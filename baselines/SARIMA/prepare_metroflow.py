"""Build an isolated, timestamp-correct MetroFlow copy for SARIMA/SARIMAX.

Run from the repository root: python -m baselines.SARIMA.prepare_metroflow
Original data, preprocessing scripts and other models are never overwritten.
"""

from argparse import ArgumentParser
import json
from pathlib import Path

import numpy as np
import pandas as pd


TARGETS = ['inFlow', 'outFlow', 'CinFlow', 'HBOinFlow', 'NHBinFlow',
           'CoutFlow', 'HBOoutFlow', 'NHBoutFlow']
WEATHER = ['temperature_2m', 'rain', 'wind_speed_10m']
# Already identified in the existing raw-data repair script. Unlike that
# script, all replacement references here are strictly earlier dates.
ANOMALOUS_DATES = (20170504, 20170508, 20170509, 20170616, 20170627, 20170628)
STEPS_PER_DAY = 102
PREPARATION_VERSION = 1


def repair_training_days(flow, dates, train_end, anomalous_dates=ANOMALOUS_DATES):
    """Repair complete anomalous TRAINING days from available past days only.

    Prefer the latest two valid occurrences of the same weekday. If no such
    dates exist at the beginning of the record, use up to seven prior valid
    days at the same time of day. Validation/test labels are never replaced.
    """
    day_values = flow.reshape(len(dates), STEPS_PER_DAY, *flow.shape[1:])
    flagged = {pd.Timestamp(str(day)).normalize() for day in anomalous_dates}
    repairs = []
    for i, day in enumerate(dates):
        if day not in flagged or (i+1)*STEPS_PER_DAY > train_end:
            continue
        earlier = [j for j in range(i) if dates[j] not in flagged]
        matching = [j for j in earlier if dates[j].dayofweek == day.dayofweek]
        references = matching[-2:] if matching else earlier[-7:]
        if not references:
            raise ValueError(f'No past reference data available to repair {day.date()}.')
        day_values[i] = np.rint(day_values[references].mean(axis=0))
        repairs.append({'date': str(day.date()),
                        'reference_dates': [str(dates[j].date()) for j in references]})
    return repairs


def prepare_metroflow(source_dir='datasets/raw_data/MetroFlow',
                      output_dir='datasets/MetroFlow_SARIMA',
                      description_path='datasets/MetroFlow/desc.json'):
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f'{output_dir} already exists; choose a new output directory.')
    with Path(description_path).open(encoding='utf-8') as stream:
        description = json.load(stream)
    ratios = description['regular_settings']['TRAIN_VAL_TEST_RATIO']
    if len(ratios) != 3 or min(ratios) <= 0 or not np.isclose(sum(ratios), 1):
        raise ValueError('MetroFlow needs three positive split ratios summing to one.')

    stations = pd.read_csv(source_dir/'stationInfo.csv', skipinitialspace=True)['stationID']
    if stations.duplicated().any():
        raise ValueError('Station identifiers must be unique.')
    columns = ['date', 'startTime', 'station', *TARGETS]
    dtypes = {column: 'float32' if column in TARGETS else 'int32' for column in columns}
    # Bypass the old interpolated CSV: its global timeslot grouping mixes dates.
    frame = pd.read_csv(source_dir/'metroData_InOutFlow.csv', skipinitialspace=True,
                        usecols=columns, dtype=dtypes)
    clock = frame['startTime'].to_numpy()
    hours, minutes, seconds = clock//10000, clock//100 % 100, clock % 100
    if np.any((hours < 6) | (hours > 22) | (minutes >= 60) | (minutes % 10 != 0) | (seconds != 0)):
        raise ValueError('Expected MetroFlow clocks from 06:00 to 22:50 every ten minutes.')
    stamp = (pd.to_datetime(frame['date'].astype(str), format='%Y%m%d')
             + pd.to_timedelta(hours.astype('int64')*3600+minutes*60, unit='s'))
    timestamps = pd.DatetimeIndex(stamp.unique()).sort_values()
    days = timestamps.normalize().unique()
    expected = pd.DatetimeIndex(np.concatenate([
        pd.date_range(day+pd.Timedelta(hours=6), periods=STEPS_PER_DAY, freq='10min').to_numpy()
        for day in days]))
    if not timestamps.equals(expected) or not days.equals(pd.date_range(days[0], days[-1], freq='D')):
        raise ValueError('MetroFlow needs 102 complete operating slots on consecutive calendar days.')
    time_index = timestamps.get_indexer(stamp)
    node_index = pd.Index(stations).get_indexer(frame['station'])
    if (node_index < 0).any():
        raise ValueError('Traffic CSV contains stations missing from stationInfo.csv.')
    encoded = time_index*len(stations)+node_index
    if len(frame) != len(timestamps)*len(stations) or len(np.unique(encoded)) != len(frame):
        raise ValueError('Expected exactly one traffic row per real timestamp and station.')
    flow = np.empty((len(timestamps), len(stations), len(TARGETS)), dtype=np.float32)
    flow[time_index, node_index] = frame[TARGETS].to_numpy(dtype=np.float32)
    if not np.isfinite(flow).all() or (flow < 0).any():
        raise ValueError('Traffic values must be finite and nonnegative.')
    del frame, encoded, time_index, node_index, stamp
    valid_len, test_len = int(len(flow)*ratios[1]), int(len(flow)*ratios[2])
    train_end = len(flow)-valid_len-test_len
    repairs = repair_training_days(flow, days, train_end)

    weather = pd.read_csv(source_dir/'MetaData'/'shanghai_weatherHourly.csv')
    weather.index = pd.to_datetime(weather['date'])
    weather = weather[WEATHER].sort_index()
    if weather.index.has_duplicates:
        raise ValueError('Weather timestamps must be unique.')
    # Historical inputs use the latest available hourly observation, without
    # interpolation against future hours or backward filling.
    weather_values = weather.reindex(timestamps, method='ffill').to_numpy()
    if not np.isfinite(weather_values).all():
        raise ValueError('Weather observations do not cover the traffic time range.')
    for feature, bins in enumerate(([20, 30, 35], [0.1, 2.5, 8.0], [5.5, 10.8, 17.1])):
        weather_values[:, feature] = np.digitize(weather_values[:, feature], bins)

    calendar = pd.read_csv(source_dir/'MetaData'/'workday_calendar.csv')
    calendar.index = pd.to_datetime(calendar['date'].astype(str), format='%Y%m%d')
    if calendar.index.has_duplicates:
        raise ValueError('Calendar dates must be unique.')
    workday = calendar['isWorkday'].reindex(timestamps.normalize()).to_numpy()
    if not np.isin(workday, [0, 1]).all():
        raise ValueError('Workday calendar does not cover all traffic dates.')

    output_dir.mkdir(parents=True, exist_ok=False)
    shape = (len(timestamps), len(stations), 14)
    output = np.memmap(output_dir/'data.dat', mode='w+', dtype='float32', shape=shape)
    output[..., :8] = flow
    output[..., 8:11] = weather_values[:, None, :]
    output[..., 11] = ((timestamps.hour*60+timestamps.minute)/1440).to_numpy()[:, None]
    output[..., 12] = timestamps.dayofweek.to_numpy()[:, None]/7
    output[..., 13] = workday[:, None]
    output.flush()
    del output
    np.save(output_dir/'timestamps.npy', timestamps.to_numpy(dtype='datetime64[ns]'))

    description.update(name=output_dir.name, source_dataset='MetroFlow', shape=list(shape),
                       num_time_steps=shape[0], num_nodes=shape[1], num_features=shape[2],
                       has_graph=False)
    description['operating_hours'] = {'steps_per_day': STEPS_PER_DAY, 'weekly_period': 714,
                                      'start': '06:00', 'end_exclusive': '23:00',
                                      'timestamps_file': 'timestamps.npy',
                                      'reset_short_term_state_daily': True,
                                      'exclude_cross_day_windows': True}
    description['preparation'] = {'version': PREPARATION_VERSION,
                                  'traffic_source': str(source_dir/'metroData_InOutFlow.csv'),
                                  'timestamp_definition': 'date + startTime (global timeslot ignored)',
                                  'weather_alignment': 'past-only hourly forward fill, then existing level bins',
                                  'date_start': str(days[0].date()), 'date_end': str(days[-1].date()),
                                  'train_steps': train_end, 'valid_steps': valid_len, 'test_steps': test_len,
                                  'training_day_repairs': repairs}
    (output_dir/'desc.json').write_text(json.dumps(description, ensure_ascii=False, indent=2), encoding='utf-8')
    return description


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', default='datasets/raw_data/MetroFlow')
    parser.add_argument('--output-dir', default='datasets/MetroFlow_SARIMA')
    parser.add_argument('--description', default='datasets/MetroFlow/desc.json')
    args = parser.parse_args()
    result = prepare_metroflow(args.source_dir, args.output_dir, args.description)
    print(json.dumps({'dataset': result['name'], 'shape': result['shape'],
                      'operating_hours': result['operating_hours'],
                      'training_day_repairs': result['preparation']['training_day_repairs']},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()

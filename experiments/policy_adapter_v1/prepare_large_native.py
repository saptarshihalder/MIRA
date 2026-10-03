"""Prepare official public data without fitting models or inspecting task scores."""
import csv
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'artifacts/runs/large_native_raw'
OUT = ROOT / 'artifacts/runs/large_native_data'


def checksum(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decode(line, rule, year):
    layout = rule['layouts'][str(year)]
    field = line[layout['start'] - 1:layout['start'] - 1 + layout['width']].strip()
    if not field:
        return np.nan
    value = int(field)
    remap = rule.get('remap', {})
    if str(value) in remap:
        return float(remap[str(value)])
    if 'valid_values' in rule:
        valid = value in rule['_allowed']
    else:
        low, high = rule['valid_range']
        valid = low <= value <= high
    return float(value) if valid else np.nan


def save(name, x, y, groups, years, ids, columns):
    path = OUT / (name + '.npz')
    np.savez_compressed(path, x=np.asarray(x, dtype=np.float32), y=np.asarray(y, dtype=np.int8),
                        groups=np.asarray(groups, dtype=np.int16), year=np.asarray(years, dtype=np.int16),
                        row_id=np.asarray(ids, dtype=np.int64))
    x = np.asarray(x)
    return dict(name=name, npz=path.name, npz_sha256=checksum(path), columns=columns, rows=len(y),
                years={str(year):int(np.sum(np.asarray(years)==year)) for year in (2024,2025)},
                missing_rate={column:float(np.isnan(x[:,j]).mean()) for j,column in enumerate(columns)})


def brfss():
    schema=json.loads((ROOT/'configs/brfss_common_variables_v1.json').read_text(encoding='utf-8'))
    for rule in schema['variables']+schema['targets']+[schema['group']]:
        if 'valid_values' in rule:
            rule['_allowed']=set(rule['valid_values'])
    variables=schema['variables']
    records={target['name']:dict(x=[],y=[],groups=[],years=[],ids=[]) for target in schema['targets']}
    raw_counts={}
    for year in (2024,2025):
        with zipfile.ZipFile(RAW/f'LLCP{year}ASC.zip') as archive:
            members=[member for member in archive.namelist() if not member.endswith('/')]
            if len(members)!=1:
                raise ValueError('Expected a single official ASCII member')
            with archive.open(members[0]) as source:
                count=0
                for row_id,line in enumerate(source):
                    count+=1
                    group=decode(line,schema['group'],year)
                    if not np.isfinite(group):
                        raise ValueError('Invalid state identity')
                    values=[decode(line,rule,year) for rule in variables]
                    for target in schema['targets']:
                        outcome=decode(line,target,year)
                        if not np.isfinite(outcome):
                            continue
                        record=records[target['name']]
                        record['x'].append(values)
                        record['y'].append(int(outcome==target['positive_value']))
                        record['groups'].append(int(group))
                        record['years'].append(year)
                        record['ids'].append(row_id)
                raw_counts[str(year)]=count
    return [save('brfss_'+name.lower(),columns=[v['name'] for v in variables],**record)
            for name,record in records.items()],raw_counts


ROAD_COLUMNS=['first_road_class','road_type','speed_limit','junction_detail_historic','junction_detail',
              'junction_control','second_road_class','pedestrian_crossing_human_control_historic',
              'pedestrian_crossing_physical_facilities_historic','pedestrian_crossing','light_conditions',
              'weather_conditions','road_surface_conditions','special_conditions_at_site',
              'carriageway_hazards_historic','carriageway_hazards','urban_or_rural_area','trunk_road_flag',
              'day_of_week','month','day_of_month','hour','minute']
ROAD_UNKNOWN=dict(road_type=9,speed_limit=99,junction_detail_historic=99,junction_detail=99,
                  junction_control=9,second_road_class=9,pedestrian_crossing_human_control_historic=9,
                  pedestrian_crossing_physical_facilities_historic=9,pedestrian_crossing=99,
                  weather_conditions=9,road_surface_conditions=9,special_conditions_at_site=9,
                  carriageway_hazards_historic=9,carriageway_hazards=99,urban_or_rural_area=3)


def road():
    x,y,groups,years,ids=[],[],[],[],[]
    raw_counts={}
    for year in (2024,2025):
        seen=set()
        with (RAW/f'dft-road-casualty-statistics-collision-{year}.csv').open(encoding='utf-8-sig',newline='') as source:
            for row_id,row in enumerate(csv.DictReader(source)):
                identity=row['collision_index']
                if identity in seen:
                    raise ValueError('Duplicate official collision identity')
                seen.add(identity)
                if int(row['collision_year'])!=year or row['collision_severity'] not in ('1','2','3'):
                    raise ValueError('Invalid road year or target')
                parts=row['date'].split('/')
                date_fields=dict(month=parts[1],day_of_month=parts[0])
                time_parts=row['time'].split(':')
                date_fields.update(hour=time_parts[0] if len(time_parts)==2 else '',
                                   minute=time_parts[1] if len(time_parts)==2 else '')
                values=[]
                for column in ROAD_COLUMNS:
                    raw=date_fields.get(column,row.get(column,''))
                    value=float(raw) if raw else np.nan
                    if value<0 or value==ROAD_UNKNOWN.get(column):
                        value=np.nan
                    values.append(value)
                x.append(values)
                y.append(int(row['collision_severity'] in ('1','2')))
                groups.append(int(row['police_force']))
                years.append(year)
                ids.append(row_id)
        raw_counts[str(year)]=len(seen)
    return save('road_ksi',x,y,groups,years,ids,ROAD_COLUMNS),raw_counts


if __name__=='__main__':
    if OUT.exists():
        raise ValueError('Preserve previous prepared data')
    OUT.mkdir(parents=True)
    tasks,health_counts=brfss()
    road_task,road_counts=road()
    tasks.append(road_task)
    manifest=dict(tasks=tasks,raw_records=dict(brfss=health_counts,road=road_counts),
                  unique_raw_records=sum(health_counts.values())+sum(road_counts.values()),
                  raw_sha256={path.name:checksum(path) for path in RAW.iterdir() if path.is_file()},
                  preparation_sha256=checksum(Path(__file__)),
                  brfss_schema_sha256=checksum(ROOT/'configs/brfss_common_variables_v1.json'),
                  scope='Two domains and two annual cohorts each; health targets share survey rows. No model scores used in preparation.')
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(dict(raw_records=manifest['raw_records'],unique_raw_records=manifest['unique_raw_records'],
                          tasks=[dict(name=t['name'],rows=t['rows'],columns=len(t['columns'])) for t in tasks])))

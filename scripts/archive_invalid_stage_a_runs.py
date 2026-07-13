from __future__ import annotations
import json,shutil
from datetime import datetime,timezone
from hashlib import sha256
from pathlib import Path

root=Path('reports/optimisation'); audit=json.loads((root/'existing_experiment_integrity.json').read_text())
archive=root/'invalid_runs_pre_latency_fix'; archive.mkdir(exist_ok=True)
rows=[]
for check in audit['checks']:
    if check['valid']: continue
    source=root/check['experiment_id']; target=archive/check['experiment_id']
    if target.exists(): shutil.rmtree(target)
    shutil.copytree(source,target)
    snapshot=(source/'config_snapshot.yaml').read_bytes()
    summary=json.loads((source/'summary.json').read_text())
    rows.append({'experiment_id':check['experiment_id'],'original_path':str(source),
                 'archived_path':str(target),'reason':'Pre-latency-fix run failed strict integrity validation.',
                 'integrity_failures':check['errors']+check['missing_files'],
                 'configuration_checksum':sha256(snapshot).hexdigest(),
                 'dataset_checksum':summary.get('dataset_sha256'),
                 'archived_timestamp':datetime.now(timezone.utc).isoformat()})
(archive/'archive_manifest.json').write_text(json.dumps(rows,indent=2)+'\n')
(archive/'archive_manifest.md').write_text('# Invalid Runs Before Latency Fix\n\n'+
    '\n'.join(f"- {r['experiment_id']}: {', '.join(r['integrity_failures'])}" for r in rows)+'\n')
print(json.dumps({'archived':len(rows)},indent=2))

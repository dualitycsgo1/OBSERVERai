"""Read-only runtime diagnostics. No fabricated notifications or accuracy scores."""
from pathlib import Path
import json
import pickle

from observer_model import validate_model_bundle


def ensure_model_integration():
    base = Path(__file__).resolve().parent
    model_path = base / 'models/observer_ai_model.pkl'
    status = {'parser_available': (base / 'demo_parser_with_positions.exe').is_file(),
              'model_available': model_path.is_file(), 'model_valid': False,
              'collision_assets': sorted(path.name for path in (base / 'maps').glob('*')
                                         if path.suffix == '.mesh' or path.name.endswith('_collision.npz')),
              'fallback': 'heuristics when no validated kill model or collision mesh is available'}
    if status['model_available']:
        try:
            with model_path.open('rb') as stream:
                bundle = validate_model_bundle(pickle.load(stream))
            status.update(model_valid=True, target=bundle['target'],
                          horizon_seconds=bundle['horizon_seconds'], evaluation=bundle['evaluation'])
        except Exception as error:
            status['model_error'] = str(error)
    print(json.dumps(status, indent=2))
    return status


if __name__ == '__main__':
    ensure_model_integration()

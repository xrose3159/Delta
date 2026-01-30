import json
import sys
import argparse
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from adaptive_training.processors.llmasjudge import evaluate_predictions_with_judge


def main():
    parser = argparse.ArgumentParser(description='Run LLM Judge evaluation')
    parser.add_argument('eval_records_file', type=str, help='Path to eval records JSON file')
    parser.add_argument('predictions_path', type=str, help='Path to predictions file')
    parser.add_argument('output_file', type=str, help='Path to output JSON file')
    parser.add_argument('judge_config', type=str, help='Judge configuration as JSON string')
    
    args = parser.parse_args()
    
    try:
        with open(args.eval_records_file, 'r') as f:
            eval_records = json.load(f)

        judge_config = json.loads(args.judge_config)

        configured_max_len = judge_config.get('max_model_len', 70000)
        eval_data, judge_stats = evaluate_predictions_with_judge(
            predictions_path=args.predictions_path,
            eval_records=eval_records,
            model_path=judge_config['model_path'],
            tensor_parallel_size=judge_config['tensor_parallel_size'],
            gpu_memory_utilization=judge_config['gpu_memory_utilization'],
            temperature=judge_config['temperature'],
            max_tokens=judge_config['max_tokens'],
            max_model_len=configured_max_len,
        )
        with open(args.output_file, 'w') as f:
            json.dump({'eval_data': eval_data, 'judge_stats': judge_stats}, f, ensure_ascii=False)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
if __name__ == '__main__':
    main()
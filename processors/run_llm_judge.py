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
        print("🚀 LLM Judge subprocess started", flush=True)
        print(f"📁 Eval records: {args.eval_records_file}", flush=True)
        print(f"📁 Predictions: {args.predictions_path}", flush=True)
        print(f"📁 Output: {args.output_file}", flush=True)


        print("📖 Reading eval records...", flush=True)
        with open(args.eval_records_file, 'r') as f:
            eval_records = json.load(f)
        print(f"✅ Loaded {len(eval_records)} eval records", flush=True)


        print("⚙️  Parsing judge config...", flush=True)
        judge_config = json.loads(args.judge_config)
        print(f"📊 Judge config: {judge_config}", flush=True)

        configured_max_len = judge_config.get('max_model_len', 70000)


        print("🔥 Starting LLM Judge evaluation...", flush=True)
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
        print("✅ LLM Judge evaluation completed", flush=True)


        print(f"💾 Saving results to {args.output_file}...", flush=True)
        with open(args.output_file, 'w') as f:
            json.dump({'eval_data': eval_data, 'judge_stats': judge_stats}, f, ensure_ascii=False)

        print("✅ LLM Judge completed successfully", flush=True)

    except Exception as e:
        print(f"❌ Fatal error in LLM Judge subprocess: {e}", flush=True)
        import traceback
        print("📋 Full traceback:", flush=True)
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

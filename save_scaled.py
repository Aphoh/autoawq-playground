import argparse
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default='mistralai/Mistral-7B-v0.1')
    parser.add_argument("--quant_path", type=str, default='mistral-awq')
    parser.add_argument("--zero_point", action='store_true')
    parser.add_argument("--q_group_size", type=int, default=128)
    parser.add_argument("--w_bit", type=int, default=6)
    args = parser.parse_args()

    quant_config = { "zero_point": args.zero_point, "q_group_size": args.q_group_size, "w_bit": args.w_bit, "version": "GEMM" }

    # Load model
    # NOTE: pass safetensors=True to load safetensors
    model = AutoAWQForCausalLM.from_pretrained(
        args.model_path, low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_cache=False)

    # Quantize
    # NOTE: We avoid packing weights, so you cannot use this model in AutoAWQ
    # after quantizing. The saved model is FP16 but has the AWQ scales applied.
    model.quantize(
        tokenizer,
        quant_config=quant_config,
        export_compatible=True
    )

    # Save quantized model
    model.save_quantized(args.quant_path)

if __name__ == "__main__":
    main()
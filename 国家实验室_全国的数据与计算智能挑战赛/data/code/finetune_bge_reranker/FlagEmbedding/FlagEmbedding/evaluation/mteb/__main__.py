from transformers import HfArgumentParser

from FlagEmbedding.evaluation.mteb import (
    MTEBEvalArgs, MTEBEvalModelArgs,
    MTEBEvalRunner
)


parser = HfArgumentParser((
    MTEBEvalArgs,
    MTEBEvalModelArgs
))

eval_args, model_args = parser.parse_args_into_dataclasses()
eval_args: MTEBEvalArgs
model_args: MTEBEvalModelArgs

runner = MTEBEvalRunner(
    eval_args=eval_args,
    model_args=model_args
)

runner.run()

from FlagEmbedding.abc.evaluation import AbsEvalRunner

from .data_loader import MSMARCOEvalDataLoader


class MSMARCOEvalRunner(AbsEvalRunner):
    def load_data_loader(self) -> MSMARCOEvalDataLoader:
        data_loader = MSMARCOEvalDataLoader(
            eval_name=self.eval_args.eval_name,
            dataset_dir=self.eval_args.dataset_dir,
            cache_dir=self.eval_args.cache_path,
            token=self.eval_args.token,
            force_redownload=self.eval_args.force_redownload,
        )
        return data_loader

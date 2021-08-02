import logging
from dipy.workflows.workflow import Workflow
from dipy.atlasing.bundles import compute_atlas_bundle


class DiscreteBundleAtlasFlow(Workflow):
    @classmethod
    def get_short_name(cls):
        return 'bdaf'

    def run(self, in_dir, subjects=None, group=None, mid_path='',
            bundle_names=None, model_bundle_dir=None, out_dir=None,
            merge_out=False, save_temp=False, n_stream_min=10,
            n_stream_max=5000, n_point=20, distance='mdf',
            comb_method='rlap', skip_pairs=False):
        """Workflow of discrete bundle atlas generation.

        Given several segmented bundles as input, compute the atlas by
        combining the bundles pairwise.

        Parameters
        ----------
        in_dir : str
            Input folder.
        subjects : str, optional
            Path to a BIDS-like participants.tsv file with the IDs of the
            subjects to be processed. If None, all folders in ``in_dir`` are
            are considered as subjects.
        group : str, optional
            Label to select a subject group when the tsv file defining subjects
            has a ``group`` column. If None, all subjects are processed.
        mid_path : str, optional
            Intermediate path between ``in_dir`` and bundle files. Default is
            ''.
        bundle_names : str, optional
            Path to a tsv file with the names of the bundles to be processed.
            If None, all trk files of the first subject will be considered as
            bundle_names.
        model_bundle_dir : str, optional
            Directory with model bundles to be used as a reference to move all
            bundles to a common space. If None, bundles are assumed to be in
            the same space and no registration is performed.
        out_dir : str, optional
            Output directory. If None, the current working directory is used.
        merge_out : boolean, optional
            If True the resulting atlases of all bundles are combined into a
            single file. Default is False.
        save_temp : boolean, optional
            If True the intermediate results of each tree level are saved in a
            temp folder in trk and png formats. Default is False.
        n_stream_min : int, optional
            Bundles with less than ``n_stream_min`` streamlines wont be
            processed. Default is 10.
        n_stream_max : int, optional
            Bundles with more than ``n_stream_max`` streamlines are cropped to
            have that number and speed up the computation. Default is 5000.
        n_point : int, optional
            All streamlines are set to have ``n_point`` points. Default is 20.
        distance : str, optional
            Distance metric to be used to combine bundles. Default is 'mdf'.
            The 'mdf_se' distance uses only start/end points of streamlines.
        comb_method : str, optional
            Method used to combine each bundle pair. Default is 'rlap'.
        skip_pairs : boolean, optional
            If true bundle combination steps are randomly skipped. This helps
            to obtain a sharper result. Default is False.
        """
        logging.info("workflow running")

        _, _ = compute_atlas_bundle(in_dir=in_dir,
                                    subjects=subjects,
                                    group=group,
                                    mid_path=mid_path,
                                    bundle_names=bundle_names,
                                    model_bundle_dir=model_bundle_dir,
                                    out_dir=out_dir,
                                    merge_out=merge_out,
                                    save_temp=save_temp,
                                    n_stream_min=n_stream_min,
                                    n_stream_max=n_stream_max,
                                    n_point=n_point,
                                    distance=distance,
                                    comb_method=comb_method,
                                    skip_pairs=skip_pairs)

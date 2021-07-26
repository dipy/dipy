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
            n_stream_max=5000, n_points=20, distance='mdf_se',
            comb_method='rlap', skip_pairs=False,):
        """Workflow of discrete bundle atlas generation.

        Computes a discrete atlas representative of the bundles and saves the
        results in a directory specified by ``out_dir``.

        Parameters
        ----------
        in_dir : string
            Path to the input bundle files. This path may contain
            wildcards to process multiple inputs at once.
        subjects : string, optional
            File with the list of subjects to process.
        group : string, optional
            Group label used to select subjects when a .tsv file is provided.
        mid_path : string, optional
            Intermediate path between in_dir/subject and trk files.
        bundle_names : string, optional
            File with the list of bundles to process.
        model_bundle_dir : string, optional
            Path to the model bundle files.
        out_dir : string, optional
            Output directory. (default current directory)
        merge_out : boolean, optional
            If True computed bundle atlases are merged into a single
            trk file. Default is False.
        save_temp : boolean, optional
            To save intermediate trk
        n_stream_min : int, optional
            Minimum number of streamlines per bundle.
        n_stream_max : int, optional
            Maximum number of streamlines per bundle.
        n_points : int, optional
            Number of points of each streamline.
        distance : string, optional
            Distance metric used for bundle combination.
        comb_method : string, optional
            Method used to combine each pair of bundles.
        skip_pairs : boolean, optional
            To randomply skip several steps when combining bundles.
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
                                    n_points=n_points,
                                    distance=distance,
                                    comb_method=comb_method,
                                    skip_pairs=skip_pairs)

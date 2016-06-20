from nibabel.tmpdirs import TemporaryDirectory

from dipy.data import get_data
from dipy.workflows.segment import MedianOtsuFlow


def test_force_overwrite():
    with TemporaryDirectory() as out_dir:
        data_path, _, _ = get_data('small_25')
        mo_flow = MedianOtsuFlow(output_strategy='absolute')

        # Generate the first results
        mo_flow.run(data_path, out_dir=out_dir, save_masked=True)

        # re-run with no force overwrite, should have any outputs
        mo_flow.run(data_path, out_dir=out_dir, save_masked=True)
        outputs = mo_flow.last_generated_outputs
        assert len(outputs) == 0

        # re-run with force overwrite, should have outputs
        mo_flow._force_overwrite = True
        mo_flow.run(data_path, out_dir=out_dir, save_masked=True)
        outputs = mo_flow.last_generated_outputs
        assert len(outputs) > 0

if __name__ == '__main__':
    test_force_overwrite()

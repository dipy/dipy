import sys

import pytest
import trx.trx_file_memmap as tmm

from dipy.data import get_fnames
from dipy.io.streamline import load_tractogram
from dipy.utils.tractogram import concatenate_tractogram

is_big_endian = "big" in sys.byteorder.lower()


@pytest.mark.skipif(is_big_endian, reason="Little Endian architecture required")
def test_concatenate():
    filepath_dix, _, _ = get_fnames(name="gold_standard_tracks")
    sft = load_tractogram(filepath_dix["gs.trk"], filepath_dix["gs.nii"])
    trx = tmm.load(filepath_dix["gs.trx"])
    concat = concatenate_tractogram([sft, trx])

    assert len(concat) == 2 * len(trx)
    trx.close()
    concat.close()

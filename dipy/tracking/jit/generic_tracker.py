import logging

from nibabel.streamlines.array_sequence import ArraySequence
from nibabel.streamlines.tractogram import Tractogram
import numpy as np
from tqdm import tqdm
from trx.trx_file_memmap import TrxFile

from dipy.io.stateful_tractogram import Space, StatefulTractogram

logger = logging.getLogger("GPUStreamlines")


class GenericJITTracker:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def _ngpus(self):
        if hasattr(self, "ngpus"):
            return self.ngpus
        else:
            return 1

    def _divide_chunks(self, seeds):
        global_chunk_sz = self.chunk_size * self._ngpus()
        nchunks = (seeds.shape[0] + global_chunk_sz - 1) // global_chunk_sz
        return global_chunk_sz, nchunks

    def generate_array_sequence(self, seeds):
        global_chunk_sz, nchunks = self._divide_chunks(seeds)
        buffer_size = 0
        generators = []

        with tqdm(total=seeds.shape[0]) as pbar:
            for idx in range(nchunks):
                self.propagate(
                    seeds[idx * global_chunk_sz : (idx + 1) * global_chunk_sz]
                )
                buffer_size += self.get_buffer_size()
                generators.append(self.as_generator())
                pbar.update(
                    seeds[idx * global_chunk_sz : (idx + 1) * global_chunk_sz].shape[0]
                )
        return ArraySequence((item for gen in generators for item in gen), buffer_size)

    def generate_tractogram(self, seeds, affine):
        return Tractogram(self.generate_array_sequence(seeds), affine_to_rasmm=affine)

    def generate_sft(self, seeds, ref_img):
        return StatefulTractogram(
            self.generate_array_sequence(seeds), ref_img, Space.VOX
        )

    def generate_trx(self, seeds, ref_img):
        global_chunk_sz, nchunks = self._divide_chunks(seeds)

        # Will resize by a factor of 2 if these are exceeded
        sl_len_guess = 100
        sl_per_seed_guess = 2
        n_sls_guess = sl_per_seed_guess * seeds.shape[0]

        # trx files use memory mapping
        trx_reference = TrxFile(reference=ref_img)
        trx_reference.streamlines._data = trx_reference.streamlines._data.astype(
            np.float32
        )
        trx_reference.streamlines._offsets = trx_reference.streamlines._offsets.astype(
            np.uint64
        )

        trx_file = TrxFile(
            nb_streamlines=n_sls_guess,
            nb_vertices=n_sls_guess * sl_len_guess,
            init_as=trx_reference,
        )
        offsets_idx = 0
        sls_data_idx = 0

        with tqdm(total=seeds.shape[0]) as pbar:
            for idx in range(int(nchunks)):
                self.propagate(
                    seeds[idx * global_chunk_sz : (idx + 1) * global_chunk_sz]
                )
                tractogram = Tractogram(
                    self.as_array_sequence(),
                    affine_to_rasmm=ref_img.affine,
                )
                tractogram.to_world()
                sls = tractogram.streamlines

                new_offsets_idx = offsets_idx + len(sls._offsets)
                new_sls_data_idx = sls_data_idx + len(sls._data)

                if (
                    new_offsets_idx > trx_file.header["NB_STREAMLINES"]
                    or new_sls_data_idx > trx_file.header["NB_VERTICES"]
                ):
                    logger.info("TRX resizing...")
                    trx_file.resize(
                        nb_streamlines=new_offsets_idx * 2,
                        nb_vertices=new_sls_data_idx * 2,
                    )

                # TRX uses memmaps here
                trx_file.streamlines._data[sls_data_idx:new_sls_data_idx] = sls._data
                trx_file.streamlines._offsets[offsets_idx:new_offsets_idx] = (
                    sls_data_idx + sls._offsets
                )
                trx_file.streamlines._lengths[offsets_idx:new_offsets_idx] = (
                    sls._lengths
                )

                offsets_idx = new_offsets_idx
                sls_data_idx = new_sls_data_idx
                pbar.update(
                    seeds[idx * global_chunk_sz : (idx + 1) * global_chunk_sz].shape[0]
                )
        trx_file.resize()

        return trx_file

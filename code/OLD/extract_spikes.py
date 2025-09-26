import numpy as np
from pathlib import Path
import shutil
import glob

import spikeinterface as si
import spikeinterface.extractors as se
from spikeinterface.exporters import export_to_phy

import argparse


def extract_spikes(sorting_folder, results_folder):
    session_folder = Path(str(sorting_folder).split("_sorted")[0])
    scratch_folder = Path("/scratch")

    # At some point the directory structure changed- handle different cases.
    ecephys_folder = session_folder / "ecephys_clipped"
    if ecephys_folder.is_dir():
        ecephys_compressed_folder = session_folder / "ecephys_compressed"
    else:
        ecephys_folder = session_folder / "ecephys" / "ecephys_clipped"
        ecephys_compressed_folder = (
            session_folder / "ecephys" / "ecephys_compressed"
        )
    print(f"ecephys folder: {ecephys_folder}")
    print(f"ecephys compressed folder: {ecephys_compressed_folder}")

    sorting_curated_folder = sorting_folder / "sorting_precurated"
    postprocessed_folder = sorting_folder / "postprocessed"

    # extract stream names

    stream_names, stream_ids = se.get_neo_streams("openephys", ecephys_folder)

    neuropix_streams = [s for s in stream_names if "Neuropix" in s]
    probe_names = [s.split(".")[1].split("-")[0] for s in neuropix_streams]

    RMS_WIN_LENGTH_SECS = 3
    WELCH_WIN_LENGTH_SAMPLES = 1024

    for idx, stream_name in enumerate(neuropix_streams):
        if "-LFP" in stream_name:
            continue

        print(stream_name)

        probe_name = probe_names[idx]

        output_folder = Path(results_folder) / probe_name

        if not output_folder.is_dir():
            output_folder.mkdir()

        print("Loading sorting analyzer...")
        analyzer_folder = (
            postprocessed_folder / f"experiment1_{stream_name}_recording1.zarr"
        )
        if analyzer_folder.is_dir():
            analyzer = si.load_sorting_analyzer(analyzer_folder)
        else:
            analyzer = si.load_sorting_analyzer_or_waveforms(
                postprocessed_folder / f"experiment1_{stream_name}_recording1"
            )

        phy_folder = scratch_folder / f"{postprocessed_folder.parent.name}_phy"

        print("Exporting to phy format...")
        export_to_phy(
            analyzer,
            output_folder=phy_folder,
            compute_pc_features=False,
            remove_if_exists=True,
            copy_binary=False,
            dtype="int16",
        )

        spike_locations = analyzer.get_extension("spike_locations").get_data()
        spike_depths = spike_locations["y"]

        print("Converting data...")
        # convert clusters and squeeze
        clusters = np.load(phy_folder / "spike_clusters.npy")
        np.save(
            phy_folder / "spike_clusters.npy",
            np.squeeze(clusters.astype("uint32")),
        )

        # convert times and squeeze
        times = np.load(phy_folder / "spike_times.npy")
        np.save(
            phy_folder / "spike_times.npy",
            np.squeeze(times / 30000.0).astype("float64"),
        )

        # convert amplitudes and squeeze
        amps = np.load(phy_folder / "amplitudes.npy")
        np.save(
            phy_folder / "amplitudes.npy",
            np.squeeze(-amps / 1e6).astype("float64"),
        )

        # save depths and channel inds
        np.save(phy_folder / "spike_depths.npy", spike_depths)
        np.save(
            phy_folder / "channel_inds.npy",
            np.arange(analyzer.get_num_channels()),
            dtype="int",
        )

        # save templates
        cluster_channels = []
        cluster_peakToTrough = []
        cluster_waveforms = []
        num_chans = []

        template_ext = analyzer.get_extension("templates")
        templates = template_ext.get_templates()
        channel_locs = analyzer.get_channel_locations()

        for unit_idx, unit_id in enumerate(analyzer.unit_ids):
            waveform = templates[unit_idx, :, :]
            peak_channel = np.argmax(np.max(waveform, 0) - np.min(waveform, 0))
            peak_waveform = waveform[:, peak_channel]
            peakToTrough = (
                np.argmax(peak_waveform) - np.argmin(peak_waveform)
            ) / 30000.0
            cluster_channels.append(
                peak_channel
            )  # int(channel_locs[peak_channel,1] / 10))
            cluster_peakToTrough.append(peakToTrough)
            cluster_waveforms.append(waveform)

        np.save(
            phy_folder / "cluster_peakToTrough.npy",
            np.array(cluster_peakToTrough),
        )
        np.save(
            phy_folder / "cluster_waveforms.npy", np.stack(cluster_waveforms)
        )
        np.save(
            phy_folder / "cluster_channels.npy", np.array(cluster_channels)
        )

        # rename files
        _FILE_RENAMES = [  # file_in, file_out
            ("channel_positions.npy", "channels.localCoordinates.npy"),
            ("channel_inds.npy", "channels.rawInd.npy"),
            ("cluster_peakToTrough.npy", "clusters.peakToTrough.npy"),
            ("cluster_channels.npy", "clusters.channels.npy"),
            ("cluster_waveforms.npy", "clusters.waveforms.npy"),
            ("spike_clusters.npy", "spikes.clusters.npy"),
            ("amplitudes.npy", "spikes.amps.npy"),
            ("spike_depths.npy", "spikes.depths.npy"),
            ("spike_times.npy", "spikes.times.npy"),
        ]

        input_directory = phy_folder

        for names in _FILE_RENAMES:
            old_name = input_directory / names[0]
            new_name = output_folder / names[1]
            shutil.copyfile(old_name, new_name)

        # save quality metrics
        qm = analyzer.get_extension("quality_metrics")

        qm_data = qm.get_data()

        qm_data.index.name = "cluster_id"
        qm_data["cluster_id.1"] = qm_data.index.values

        qm_data.to_csv(output_folder / "clusters.metrics.csv")


if __name__ == "__main__":
    # set up directories
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        dest="sorting_folder",
        default=None,
        help="Sorted Folder to use as baseline",
    )
    args = parser.parse_args()

    # set up directories
    if args.sorting_folder is None:
        sorting_folder = Path(glob.glob("/data/ecephys_*sorted*")[0])
        results_folder = Path("/results/")
    else:
        sorting_folder = Path("/data/") / args.sorting_folder
        session_name = Path(str(sorting_folder).split("_sorted")[0]).name
        results_folder = Path("/results/") / session_name
        results_folder.mkdir(exist_ok=True)

    extract_spikes(sorting_folder, results_folder)


"""Load all data."""
# Spikes
self.spike_samples, self.spike_times = self._load_spike_samples()
ns, = self.n_spikes, = self.spike_times.shape

# Make sure the spike times are increasing.
if not np.all(np.diff(self.spike_times) >= 0):
    raise ValueError("The spike times must be increasing.")

# Spike amplitudes.
self.amplitudes = self._load_amplitudes()
if self.amplitudes is not None:
    assert self.amplitudes.shape == (ns,)

# Spike templates.
self.spike_templates = self._load_spike_templates()
assert self.spike_templates.shape == (ns,)

# Unique template ids.
self.template_ids = np.unique(self.spike_templates)

# Spike clusters.
self.spike_clusters = self._load_spike_clusters()
assert self.spike_clusters.shape == (ns,)

# Unique cluster ids.
self.cluster_ids = np.unique(self.spike_clusters)

# Spike reordering.
self.spike_times_reordered = self._load_spike_reorder()
if self.spike_times_reordered is not None:
    assert self.spike_times_reordered.shape == (ns,)

# Channels.
self.channel_mapping = self._load_channel_map()
self.n_channels = nc = self.channel_mapping.shape[0]
if self.n_channels_dat:
    assert np.all(self.channel_mapping <= self.n_channels_dat - 1)

# Channel positions.
self.channel_positions = self._load_channel_positions()
assert self.channel_positions.shape == (nc, 2)
if not _all_positions_distinct(self.channel_positions):  # pragma: no cover
    logger.error(
        "Some channels are on the same position, please check the channel positions file.")
    self.channel_positions = linear_positions(nc)

# Channel shanks.
self.channel_shanks = self._load_channel_shanks()
assert self.channel_shanks.shape == (nc,)

# Channel probes.
self.channel_probes = self._load_channel_probes()
assert self.channel_probes.shape == (nc,)
self.probes = np.unique(self.channel_probes)
self.n_probes = len(self.probes)

# Templates.
self.sparse_templates = self._load_templates()
if self.sparse_templates is not None:
    self.n_templates, self.n_samples_waveforms, self.n_channels_loc = \
        self.sparse_templates.data.shape
    if self.sparse_templates.cols is not None:
        assert self.sparse_templates.cols.shape == (self.n_templates, self.n_channels_loc)
else:  # pragma: no cover
    self.n_templates = self.spike_templates.max() + 1
    self.n_samples_waveforms = 0
    self.n_channels_loc = 0

# Clusters waveforms
if not np.all(self.spike_clusters == self.spike_templates) and \
        self.sparse_templates.cols is None:
    self.merge_map, self.nan_idx = self.get_merge_map()
    self.sparse_clusters = self.cluster_waveforms()
    self.n_clusters = self.spike_clusters.max() + 1
else:
    self.merge_map = {}
    self.nan_idx = []
    self.sparse_clusters = self.sparse_templates
    self.n_clusters = self.spike_templates.max() + 1

# Spike waveforms (optional, otherwise fetched from raw data as needed).
self.spike_waveforms = self._load_spike_waveforms()

# Whitening.
try:
    self.wm = self._load_wm()
except IOError:
    logger.debug("Whitening matrix file not found.")
    self.wm = np.eye(nc)
assert self.wm.shape == (nc, nc)
try:
    self.wmi = self._load_wmi()
except IOError:
    logger.debug("Whitening matrix inverse file not found, computing it.")
    self.wmi = self._compute_wmi(self.wm)
assert self.wmi.shape == (nc, nc)

# Similar templates.
self.similar_templates = self._load_similar_templates()
assert self.similar_templates.shape == (self.n_templates, self.n_templates)

# Traces and duration.
self.traces = self._load_traces(self.channel_mapping)
if self.traces is not None:
    self.duration = self.traces.duration
else:
    self.duration = self.spike_times[-1]
if self.spike_times[-1] > self.duration:  # pragma: no cover
    logger.warning(
        "There are %d/%d spikes after the end of the recording.",
        np.sum(self.spike_times > self.duration), self.n_spikes)

# Features.
self.sparse_features = self._load_features()
self.features = self.sparse_features.data if self.sparse_features else None
if self.sparse_features is not None:
    self.n_features_per_channel = self.sparse_features.data.shape[2]

# Template features.
self.sparse_template_features = self._load_template_features()
self.template_features = (
    self.sparse_template_features.data if self.sparse_template_features else None)

# Spike attributes.
self.spike_attributes = self._load_spike_attributes()

# Metadata.
self.metadata = self._load_metadata()
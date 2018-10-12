# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
import tensorflow as tf
import numpy as np
import scipy.misc
import os

try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO  # Python 3.x


class TBLogger(object):

    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.writers = {}

    def get_path(self, sub_dir):
        return os.path.join(self.log_dir, sub_dir) if sub_dir else self.log_dir

    def _get_writer(self, sub_dir):
        """Create a summary writer logging to log_dir."""
        path = self.get_path(sub_dir)
        writer = self.writers.get(path)
        if writer is None:
            self.writers[path] = tf.summary.FileWriter(path)

        return self.writers[path]

    def scalar_summary(self, sub_dir, tag, value, step):
        """Log a scalar variable."""
        writer = self._get_writer(sub_dir)
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
        writer.add_summary(summary, step)

    def image_summary(self, sub_dir, tag, images, step):
        """Log a list of images."""

        writer = self._get_writer(sub_dir)
        img_summaries = []
        for i, img in enumerate(images):
            # Write the image to a string
            try:
                s = StringIO()
            except:
                s = BytesIO()
            scipy.misc.toimage(img).save(s, format="png")

            # Create an Image object
            img_sum = tf.Summary.Image(encoded_image_string=s.getvalue(),
                                       height=img.shape[0],
                                       width=img.shape[1])
            # Create a Summary value
            img_summaries.append(tf.Summary.Value(tag='%s/%d' % (tag, i), image=img_sum))

        # Create and write Summary
        summary = tf.Summary(value=img_summaries)
        writer.add_summary(summary, step)

    def histo_summary(self, sub_dir, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""
        writer = self._get_writer(sub_dir)

        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = tf.HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = tf.Summary(value=[tf.Summary.Value(tag=tag, histo=hist)])
        writer.add_summary(summary, step)
        writer.flush()

    def flush(self, sub_dir):
        writer = self._get_writer(sub_dir)
        writer.flush()

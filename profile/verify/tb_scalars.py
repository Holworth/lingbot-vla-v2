"""Read scalars from TensorBoard event files without EventAccumulator (tensorboard 2.16 + NumPy 2 fails to import it).

    python tb_scalars.py <run_dir or event file> [tag ...]

Event files are TFRecord: [u64 length][u32 crc][Event proto][u32 crc]; scalars from SummaryWriter.add_scalar are
float32 simple_value. Returns {tag: {step: value}}; for a step logged more than once the last value wins.
"""
import glob
import os
import struct
import sys

from tensorboard.compat.proto import event_pb2


def read_scalars(path, tags=None):
    files = sorted(glob.glob(os.path.join(path, "events.out.tfevents.*"))) if os.path.isdir(path) else [path]
    out = {}
    for f in files:
        data = open(f, "rb").read()
        pos = 0
        while pos + 12 <= len(data):
            (n,) = struct.unpack_from("<Q", data, pos)
            if pos + 12 + n + 4 > len(data):
                break  # truncated last record (process killed while writing)
            ev = event_pb2.Event.FromString(data[pos + 12:pos + 12 + n])
            pos += 12 + n + 4
            for v in ev.summary.value:
                if tags and v.tag not in tags:
                    continue
                if v.HasField("simple_value"):
                    out.setdefault(v.tag, {})[ev.step] = v.simple_value
                elif v.HasField("tensor") and v.tensor.float_val:
                    out.setdefault(v.tag, {})[ev.step] = v.tensor.float_val[0]
    return out


if __name__ == "__main__":
    s = read_scalars(sys.argv[1], sys.argv[2:] or None)
    for tag, d in sorted(s.items()):
        steps = sorted(d)
        print(f"{tag}: {len(d)} steps {steps[0]}-{steps[-1]}; first {[(k, d[k]) for k in steps[:3]]}")

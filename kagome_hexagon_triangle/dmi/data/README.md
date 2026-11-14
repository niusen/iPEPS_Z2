# Runtime data

`logs/` contains preserved local run logs and historical PID records. PID files
from completed runs do not indicate a live process. New server launchers also
write their log/PID files here.

Tensor checkpoints and scientific result tables remain at their original
`kagome_hexagon_triangle/results/` locations so existing runs can be resumed.
Exclude this `data/` folder when uploading source code to the server.

# Hydraulic TIN builder

These sources are vendored from the sibling `hydraulic_triangular_irregular_networks`
project, including its linear-time local relief analysis and classic blob-worker fix.
Copyright and MIT license are preserved in LICENSE. The erosion post-process uses
this CPU Delaunay builder in a worker, once per requested frozen terrain snapshot.
The erosion solver remains on the GPU; the builder is never invoked by a simulation step.

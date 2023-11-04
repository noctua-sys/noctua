# Centralized Coordination Service

The coordination server maintains a list of currently active barriers for each site

| Site | OpID |
|------|------|
| 1    | Foo1 |
| 3    | Bar1 |

This table is shared, and accessed by each site using a defined JSONRPC interface.


## Assumptions

port 4000 = coordinator
port 4000+i = site i

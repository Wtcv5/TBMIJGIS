# Data request checklist for real BSLL shield-sticking event validation

## 1. Event log / engineering fact record
Required fields:
- event name
- confirmed event time
- confirmed event chainage
- event development window
- observed failure/stuck criterion
- engineering interpretation
- mitigation/unsticking measures
- source document or meeting minutes

## 2. Time-chainage alignment
Required fields:
- timestamp
- ring number, if available
- shield mileage / face chainage
- mapping from TSP local x to real chainage
- event-chainage anchor

## 3. TSP / advance geological prediction data
Required fields:
- voxel or profile chainage x
- y and z coordinates, if 3D TSP inversion is available
- Vp
- Vs, if available
- density, Poisson ratio, dynamic modulus, if available
- geological interpretation labels, if available
- TSP survey start/end chainage

## 4. TBM monitoring data
Required fields:
- timestamp
- shield mileage / face chainage
- shield pressure
- shield displacement or convergence-related shield reading
- total thrust
- advance rate / penetration / excavation speed
- cutterhead RPM
- torque, if available
- operating state flags, if available

## 5. TBM geometry
Required fields:
- cutterhead radius
- shield radius
- shield length
- front/middle/tail shield segment definitions
- TBM advance direction
- sampling or CAD-derived surface points, if available

## 6. Event-centred validation outputs
After receiving real data, generate:
- event-aligned TSP anomaly + shield-indexed readout + monitoring plot
- global TSP summary vs shield-indexed graph readout comparison
- top contributing rock-shield edge provenance table

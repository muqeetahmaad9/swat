// ============================================================================
// GOOGLE EARTH ENGINE SCRIPT — Advanced Terrain & Hydrological Features
// Purpose : the high-value predictors missing from the first export round:
//           HAND, TWI, SPI, log flow accumulation, TPI (2 scales), plan &
//           profile curvature, distance to river, drainage density, LULC
// Grid    : IDENTICAL to the 10 existing layers (30 m, EPSG:32643, same AOI)
// Run     : paste into code.earthengine.google.com, Run, then run all export
//           tasks. Download into Desktop\paper\data\raw\ alongside the others.
// ============================================================================

// ---------------------------------------------------------------------------
// SECTION 0 — PARAMETERS (identical to the feature/label scripts)
// ---------------------------------------------------------------------------
var aoi = ee.Geometry.Rectangle([71.0, 34.0, 72.9, 35.95]);  // same AOI
var outScale = 30;                                            // same grid
var outCRS   = 'EPSG:32643';                                  // same CRS
var driveFolder = 'GEE_Swat_Flood';                           // same folder

Map.centerObject(aoi, 9);

// ---------------------------------------------------------------------------
// SECTION 1 — BASE SURFACES
// ---------------------------------------------------------------------------

// Copernicus GLO-30 DEM with restored projection (same pattern as before).
var glo30 = ee.ImageCollection('COPERNICUS/DEM/GLO30');
var demProj = glo30.first().select('DEM').projection();
var dem = glo30.select('DEM').mosaic().setDefaultProjection(demProj);

// Slope in radians (needed by TWI/SPI formulas).
var slopeDeg = ee.Terrain.slope(dem);
var slopeRad = slopeDeg.multiply(Math.PI / 180);

// MERIT Hydro: hydrologically conditioned derivatives.
var merit = ee.Image('MERIT/Hydro/v1_0_1');
var upa  = merit.select('upa');    // upstream drainage area, km^2 (flow accumulation)
var hand = merit.select('hnd');    // height above nearest drainage, m (90 m native)

// ---------------------------------------------------------------------------
// SECTION 2 — HYDROLOGICAL INDICES
// ---------------------------------------------------------------------------

// Specific catchment area per unit contour width: a_s = upa(km2->m2) / cellwidth(m).
var sca = upa.multiply(1e6).divide(930);      // MERIT cell ~3 arcsec ≈ 93 m -> width proxy

// TWI = ln( a_s / tan(slope) ); slope floored at 0.06 deg to avoid div-by-zero.
var twi = sca.divide(slopeRad.tan().max(0.001))
             .log().rename('twi');

// SPI = ln( a_s * tan(slope) + 1 )  (log-scaled stream power).
var spi = sca.multiply(slopeRad.tan()).add(1)
             .log().rename('spi');

// log flow accumulation (heavy-tailed -> log).
var logFacc = upa.add(1).log().rename('log_facc');

// HAND, renamed for the stack (90 m native — state this in Methods).
var handOut = hand.rename('hand');

// ---------------------------------------------------------------------------
// SECTION 3 — TERRAIN MORPHOMETRY
// ---------------------------------------------------------------------------

// TPI at two scales: cell elevation minus neighbourhood mean.
var tpi300 = dem.subtract(dem.reduceNeighborhood({
  reducer: ee.Reducer.mean(), kernel: ee.Kernel.circle(300, 'meters')
})).rename('tpi300');
var tpi1000 = dem.subtract(dem.reduceNeighborhood({
  reducer: ee.Reducer.mean(), kernel: ee.Kernel.circle(1000, 'meters')
})).rename('tpi1000');

// Plan & profile curvature computed natively (no external library):
// first and second partial derivatives of the surface via ee.Image.gradient(),
// then the standard curvature formulas (Zevenbergen & Thorne style).
var g1 = dem.gradient();                       // first derivatives
var p = g1.select('x');                        // dz/dx
var q = g1.select('y');                        // dz/dy
var zxx = p.gradient().select('x');            // d2z/dx2
var zxy = p.gradient().select('y');            // d2z/dxdy
var zyy = q.gradient().select('y');            // d2z/dy2

// Profile curvature: curvature along the flow direction (flow acceleration).
var profCurv = ee.Image().expression(
  '-(zxx*p*p + 2*zxy*p*q + zyy*q*q) / ((p*p + q*q) * pow(1 + p*p + q*q, 1.5) + 1e-9)',
  {p: p, q: q, zxx: zxx, zxy: zxy, zyy: zyy}).rename('prof_curv');

// Plan curvature: curvature across the slope (flow convergence/divergence).
var planCurv = ee.Image().expression(
  '-(zxx*q*q - 2*zxy*p*q + zyy*p*p) / (pow(p*p + q*q, 1.5) + 1e-9)',
  {p: p, q: q, zxx: zxx, zxy: zxy, zyy: zyy}).rename('plan_curv');

// ---------------------------------------------------------------------------
// SECTION 4 — RIVER DISTANCE & DRAINAGE DENSITY
// ---------------------------------------------------------------------------

// Stream network: cells draining more than 0.5 km^2.
var streams = upa.gt(0.5);

// Euclidean distance to the nearest stream cell, in metres.
var distRiver = streams.selfMask()
  .fastDistanceTransform(1024).sqrt()
  .multiply(ee.Image.pixelArea().sqrt())
  .rename('dist_river');

// Drainage density: fraction of stream cells within a 1 km radius
// (proportional to km of channel per km^2).
var dd = streams.reduceNeighborhood({
  reducer: ee.Reducer.mean(), kernel: ee.Kernel.circle(1000, 'meters')
}).rename('drainage_density');

// ---------------------------------------------------------------------------
// SECTION 5 — LAND COVER (categorical; nearest-neighbour resampling)
// ---------------------------------------------------------------------------

// ESA WorldCover v200 (2021), 10 m -> exported on the 30 m grid.
// Preprocessing pipeline treats it as categorical (add to CONFIG['categorical']).
var lulc = ee.ImageCollection('ESA/WorldCover/v200').first()
  .select('Map').rename('lulc');

// ---------------------------------------------------------------------------
// SECTION 6 — VISUAL CHECKS + EXPORTS
// ---------------------------------------------------------------------------
Map.addLayer(handOut.clip(aoi), {min: 0, max: 50}, 'HAND', false);
Map.addLayer(twi.clip(aoi), {min: 2, max: 20}, 'TWI', false);
Map.addLayer(distRiver.clip(aoi), {min: 0, max: 3000}, 'dist river', false);

function exportLayer(image, name) {
  Export.image.toDrive({
    image: image.toFloat().clip(aoi),
    description: name, fileNamePrefix: name, folder: driveFolder,
    region: aoi, scale: outScale, crs: outCRS,
    maxPixels: 1e13, fileFormat: 'GeoTIFF'
  });
}

exportLayer(handOut,  'swat_hand');
exportLayer(twi,      'swat_twi');
exportLayer(spi,      'swat_spi');
exportLayer(logFacc,  'swat_logfacc');
exportLayer(tpi300,   'swat_tpi300');
exportLayer(tpi1000,  'swat_tpi1000');
exportLayer(planCurv, 'swat_plancurv');
exportLayer(profCurv, 'swat_profcurv');
exportLayer(distRiver,'swat_distriver');
exportLayer(dd,       'swat_draindens');
exportLayer(lulc,     'swat_lulc');

print('11 export tasks queued. Run them all, download into data/raw/, ' +
      'then the preprocessing pipeline rebuilds the cube with ~29 bands.');

// ============================================================================
// GOOGLE EARTH ENGINE SCRIPT — Flood Label Extraction (Sentinel-1 change detection)
// Study area : Swat River Basin rectangle (same AOI as GEE_Swat_Data_Export.js)
// Events     : August 2022 monsoon floods + August 2025 monsoon floods
// Method     : pre-event vs flood-window VH backscatter drop, cleaned with
//              slope / HAND / permanent-water / blob filters
// Outputs    : flood_labels.tif (combined), flood_labels_2022.tif,
//              flood_labels_2025.tif — Byte rasters: 1=flood, 0=non-flood,
//              255=unknown/masked — on the EXACT grid of the 10 feature layers
// ============================================================================

// ---------------------------------------------------------------------------
// SECTION 0 — PARAMETERS
// ---------------------------------------------------------------------------

// Same AOI rectangle as the feature-export script (MUST match for grid alignment).
var aoi = ee.Geometry.Rectangle([71.0, 34.0, 72.9, 35.95]);

// Event time windows. VERIFY these against NDMA/PDMA situation reports:
// the closer the flood window brackets the actual peak, the better the labels.
// 2022: KP floods peaked around 25-27 August 2022.
var pre2022   = ['2022-06-01', '2022-07-10'];   // pre-monsoon baseline 2022
var flood2022 = ['2022-08-18', '2022-09-15'];   // flood window 2022
// 2025: KP flash floods mid-August 2025 (verify exact dates before final run).
var pre2025   = ['2025-05-01', '2025-06-15'];   // pre-monsoon baseline 2025
var flood2025 = ['2025-08-05', '2025-09-05'];   // flood window 2025

// Change-detection thresholds (tune by visual inspection of the map layers):
var DROP_DB   = 4;      // required backscatter DROP (pre - flood) in dB
var OPENWATER = -17;    // flood-window VH must also be darker than this (dB)
var SLOPE_MAX = 10;     // flooding only plausible on slopes below this (deg)
var HAND_MAX  = 20;     // and within this height above nearest drainage (m)
var MIN_BLOB  = 8;      // remove flood patches smaller than this many pixels
var BUFFER_M  = 150;    // exclusion buffer around flood pixels (label safety)

// Export grid — IDENTICAL to the feature exports.
var outScale = 30;                              // 30 m pixels
var outCRS   = 'EPSG:32643';                    // UTM 43N
var driveFolder = 'GEE_Swat_Flood';             // same Drive folder

// Center the map for visual checking.
Map.centerObject(aoi, 9);

// ---------------------------------------------------------------------------
// SECTION 1 — STATIC MASKS (terrain, drainage, permanent water)
// ---------------------------------------------------------------------------

// Copernicus GLO-30 DEM mosaic with restored projection (as in the export script).
var glo30 = ee.ImageCollection('COPERNICUS/DEM/GLO30');
var dem = glo30.select('DEM').mosaic()
  .setDefaultProjection(glo30.first().select('DEM').projection());

// Slope in degrees — floods do not stand on steep hillsides.
var slope = ee.Terrain.slope(dem);

// HAND (height above nearest drainage) from MERIT Hydro — pixels far above
// the drainage network cannot be river-flooded; removes SAR-shadow false alarms.
var hand = ee.Image('MERIT/Hydro/v1_0_1').select('hnd');

// JRC permanent water: pixels that are water most of the time are RIVER/LAKE,
// not flood — they must not be labelled as flood events.
var permWater = ee.Image('JRC/GSW1_4/GlobalSurfaceWater')
  .select('occurrence').unmask(0).gte(50);      // water >=50% of observations

// The "physically floodable" domain: gentle slope AND near the drainage network.
var floodable = slope.lte(SLOPE_MAX).and(hand.lte(HAND_MAX));

// ---------------------------------------------------------------------------
// SECTION 2 — SENTINEL-1 HELPERS
// ---------------------------------------------------------------------------

// Load Sentinel-1 IW VH scenes over the AOI for a date range AND a single
// pass direction, speckle-filtered. Change detection must NEVER mix
// ascending and descending passes: their incidence angles differ, and in
// steep terrain that alone fakes multi-dB "changes".
function s1vh(dates, pass) {
  return ee.ImageCollection('COPERNICUS/S1_GRD')
    .filterBounds(aoi)                                        // scenes over AOI
    .filterDate(dates[0], dates[1])                           // inside the window
    .filter(ee.Filter.eq('instrumentMode', 'IW'))             // standard land mode
    .filter(ee.Filter.eq('orbitProperties_pass', pass))       // ONE pass direction
    .filter(ee.Filter.listContains(
        'transmitterReceiverPolarisation', 'VH'))             // need VH band
    .select('VH')                                             // keep VH only
    .map(function (img) {                                     // per-scene cleanup
      return img.focalMedian(50, 'circle', 'meters')          // light speckle filter
                .copyProperties(img, img.propertyNames());    // keep metadata
    });
}

// Change detection for ONE event and ONE pass direction.
function floodMaskOnePass(preDates, floodDates, pass, label) {
  var pre = s1vh(preDates, pass);                             // baseline collection
  var flo = s1vh(floodDates, pass);                           // flood collection
  print('S1 scenes ' + label + ' ' + pass + ': pre =',        // sanity counts
        pre.size(), ' flood =', flo.size());
  var preMed  = pre.median();                                 // typical dry state (dB)
  var floMin  = flo.min();                                    // darkest flood state (dB)
  var drop    = preMed.subtract(floMin);                      // backscatter drop (dB)
  var flooded = drop.gte(DROP_DB)                             // big drop ...
      .and(floMin.lte(OPENWATER))                             // ... to open-water levels
      .and(floodable)                                         // physically plausible
      .and(permWater.not());                                  // not permanent water
  // Remove tiny speckle blobs: keep only connected patches >= MIN_BLOB pixels.
  var blobs = flooded.selfMask()
      .connectedPixelCount(MIN_BLOB + 1, false);              // count neighbours
  flooded = flooded.updateMask(blobs.gte(MIN_BLOB))           // drop small blobs
      .unmask(0);                                             // holes back to 0
  Map.addLayer(drop.clip(aoi), {min: 0, max: 8},
               'VH drop ' + label + ' ' + pass, false);
  return flooded;                                             // 1 = flood, 0 = not
}

// Build one event's flood mask: detect PER PASS, then take the union
// (a pixel is flood if either viewing geometry saw it flooded).
function floodMask(preDates, floodDates, label) {
  var asc = floodMaskOnePass(preDates, floodDates, 'ASCENDING',  label);
  var des = floodMaskOnePass(preDates, floodDates, 'DESCENDING', label);
  var flooded = asc.or(des);                                  // union of geometries
  Map.addLayer(flooded.selfMask().clip(aoi),
               {palette: ['ff00ff']}, 'flood ' + label, false);
  return flooded.rename('flood');                             // 1 = flood, 0 = not
}

// ---------------------------------------------------------------------------
// SECTION 3 — EVENT MASKS AND FINAL LABEL RASTER
// ---------------------------------------------------------------------------

// Per-event binary flood masks.
var f2022 = floodMask(pre2022, flood2022, '2022');
var f2025 = floodMask(pre2025, flood2025, '2025');

// Combined flood evidence: flooded in EITHER event.
var floodAny = f2022.or(f2025);

// Uncertainty buffer: pixels NEAR flood but not flood are ambiguous (mixed
// pixels, receding water). Mark a ring around flood as UNKNOWN, not non-flood.
var nearFlood = floodAny.selfMask()
    .fastDistanceTransform(64).sqrt()
    .multiply(ee.Image.pixelArea().sqrt())                    // distance in metres
    .lte(BUFFER_M);                                           // within the buffer

// Assemble the 3-class label raster:
//   start UNKNOWN(255) -> set floodable dry land to 0 -> ring to 255 -> flood to 1.
var labels = ee.Image(255).byte()                             // default: unknown
    .where(floodable.and(permWater.not()), 0)                 // plausible dry = 0
    .where(nearFlood.unmask(0), 255)                          // ambiguous ring = unknown
    .where(floodAny, 1)                                       // flooded = 1
    .rename('label')
    .clip(aoi);

// NOTE: non-flood (0) is restricted to the FLOODABLE domain on purpose:
// negatives sampled from steep hillslopes would be trivially easy and would
// inflate accuracy. The model learns the hard boundary: floodable-but-dry
// vs flooded. State this design choice in the Methods section.

// Visual check layers.
Map.addLayer(labels.updateMask(labels.neq(255)).clip(aoi),
             {min: 0, max: 1, palette: ['cccccc', '0000ff']}, 'labels (0/1)');
print('Flood pixel count (combined):',
      floodAny.multiply(ee.Image.pixelArea()).divide(1e6)
        .reduceRegion({reducer: ee.Reducer.sum(), geometry: aoi,
                       scale: 300, maxPixels: 1e13}));        // flooded km^2 (approx)

// ---------------------------------------------------------------------------
// SECTION 4 — EXPORTS (same grid as the 10 feature layers)
// ---------------------------------------------------------------------------

function exportLabel(image, name) {
  Export.image.toDrive({
    image: image.toByte(),                     // Byte raster: 0/1/255
    description: name,
    fileNamePrefix: name,
    folder: driveFolder,
    region: aoi,                               // SAME region ...
    scale: outScale,                           // ... same scale ...
    crs: outCRS,                               // ... same CRS as the features
    maxPixels: 1e13,
    fileFormat: 'GeoTIFF',
    formatOptions: {noData: 255}               // declare 255 as nodata
  });
}

exportLabel(labels, 'flood_labels');           // combined (train on this first)
// Per-event labels for the temporal-transfer experiment. CRITICAL: pixels
// flooded ONLY in the other event are set to UNKNOWN (255), not non-flood —
// otherwise known-susceptible locations serve as negatives and bias the
// cross-event evaluation (reviewer point R1.10).
exportLabel(f2022.byte()
    .where(labels.eq(255), 255)                // combined unknown stays unknown
    .where(f2025.and(f2022.not()), 255),       // 2025-only flood -> unknown
    'flood_labels_2022');
exportLabel(f2025.byte()
    .where(labels.eq(255), 255)                // combined unknown stays unknown
    .where(f2022.and(f2025.not()), 255),       // 2022-only flood -> unknown
    'flood_labels_2025');

print('3 export tasks queued. Inspect the map layers FIRST (tune DROP_DB / ' +
      'OPENWATER if flood looks over/under-detected), then run the tasks. ' +
      'Download flood_labels.tif into data/labels/ and re-run the pipeline.');

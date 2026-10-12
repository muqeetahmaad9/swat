// ============================================================================
// GOOGLE EARTH ENGINE SCRIPT — Flood Susceptibility Data Preparation
// Study area : Swat River Basin, Khyber Pakhtunkhwa, Pakistan
// Outputs    : Sentinel-1 (VV, VH), Sentinel-2 composite, NDVI, NDWI, MNDWI,
//              DEM, Slope, Aspect, Rainfall — all clipped + exported as GeoTIFF
// Run in     : https://code.earthengine.google.com  (JavaScript Code Editor)
// Exports go : Google Drive folder "GEE_Swat_Flood" (change below if desired)
// ============================================================================

// ---------------------------------------------------------------------------
// SECTION 0 — USER PARAMETERS (edit these)
// ---------------------------------------------------------------------------

// Define the study area as a rectangle around the Swat River Basin
// (west longitude, south latitude, east longitude, north latitude).
var aoi = ee.Geometry.Rectangle([71.0, 34.0, 72.9, 35.95]);

// --- ALTERNATIVE 1: use the official Swat DISTRICT boundary from FAO GAUL ---
// var aoi = ee.FeatureCollection('FAO/GAUL/2015/level2')        // load global admin level-2 boundaries
//   .filter(ee.Filter.eq('ADM2_NAME', 'Swat'))                  // keep only the Swat district polygon
//   .geometry();                                                // convert the FeatureCollection to a geometry

// --- ALTERNATIVE 2: upload your own basin shapefile as an asset, then: ------
// var aoi = ee.FeatureCollection('users/YOUR_USERNAME/swat_basin').geometry();

// Start date for the Sentinel-1 / Sentinel-2 / rainfall time windows.
var startDate = '2023-01-01';

// End date for the same time windows.
var endDate   = '2024-01-01';

// Rainfall climatology period — first year included.
var rainStartYear = 1994;

// Rainfall climatology period — last year included.
var rainEndYear   = 2023;

// Google Drive folder name that will receive all the GeoTIFF exports.
var driveFolder = 'GEE_Swat_Flood';

// Output pixel size in metres for all exported rasters (common analysis grid).
var outScale = 30;

// Output coordinate reference system (UTM zone 43N covers the Swat basin).
var outCRS = 'EPSG:32643';

// Centre the interactive map on the study area at zoom level 9.
Map.centerObject(aoi, 9);

// Draw the study-area outline on the map so you can verify it visually.
Map.addLayer(aoi, {color: 'red'}, 'Study area (AOI)', false);

// ---------------------------------------------------------------------------
// SECTION 1 — DEM (Copernicus GLO-30), SLOPE and ASPECT
// ---------------------------------------------------------------------------

// Load the Copernicus GLO-30 DEM, which is stored as an ImageCollection of tiles.
var glo30 = ee.ImageCollection('COPERNICUS/DEM/GLO30');

// Grab the native projection of one tile BEFORE mosaicking (mosaics lose it).
var demProj = glo30.first().select('DEM').projection();

// Mosaic all DEM tiles into one image, keep the elevation band,
// restore the native 30 m projection, and clip to the study area.
var dem = glo30.select('DEM')                 // keep only the elevation band
  .mosaic()                                   // merge all tiles into a single image
  .setDefaultProjection(demProj)              // re-attach the native projection (needed for terrain math)
  .clip(aoi)                                  // cut the mosaic down to the study area
  .rename('elevation');                       // give the band a clean name

// Compute slope in degrees from the DEM.
var slope = ee.Terrain.slope(dem)             // per-pixel slope from elevation neighbours
  .clip(aoi)                                  // clip result to the study area
  .rename('slope');                           // clean band name

// Compute aspect in degrees clockwise from north.
var aspect = ee.Terrain.aspect(dem)           // per-pixel downslope direction
  .clip(aoi)                                  // clip result to the study area
  .rename('aspect');                          // clean band name

// Display the DEM with an elevation colour ramp (initially hidden).
Map.addLayer(dem, {min: 300, max: 5500, palette: ['006600','ffcc66','996633','ffffff']}, 'DEM', false);

// Display the slope layer in greys (initially hidden).
Map.addLayer(slope, {min: 0, max: 60}, 'Slope', false);

// ---------------------------------------------------------------------------
// SECTION 2 — SENTINEL-1 SAR (VV and VH composites)
// ---------------------------------------------------------------------------

// Load the Sentinel-1 Ground-Range-Detected archive (calibrated sigma0 in dB).
var s1 = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(aoi)                                                       // keep scenes touching the AOI
  .filterDate(startDate, endDate)                                          // keep scenes inside the time window
  .filter(ee.Filter.eq('instrumentMode', 'IW'))                            // keep Interferometric-Wide mode (standard over land)
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation','VV'))  // require the VV band
  .filter(ee.Filter.listContains('transmitterReceiverPolarisation','VH')); // require the VH band too

// Print how many Sentinel-1 scenes matched (check the Console tab).
print('Sentinel-1 scenes found:', s1.size());

// Define a light speckle filter: a 50 m radius circular median kernel.
var speckle = function(img) {                              // function applied to every scene
  return img.focalMedian(50, 'circle', 'meters')           // median filter suppresses SAR speckle noise
            .copyProperties(img, img.propertyNames());     // keep the original metadata
};

// Apply the speckle filter to every scene in the collection.
var s1Filtered = s1.map(speckle);

// Build the temporal MEDIAN composite of VV backscatter (robust to outliers).
var s1VV = s1Filtered.select('VV')            // keep the VV polarisation band
  .median()                                   // per-pixel median over the year
  .clip(aoi)                                  // clip to the study area
  .rename('VV');                              // clean band name

// Build the temporal MEDIAN composite of VH backscatter.
var s1VH = s1Filtered.select('VH')            // keep the VH polarisation band
  .median()                                   // per-pixel median over the year
  .clip(aoi)                                  // clip to the study area
  .rename('VH');                              // clean band name

// Display the VV composite (dark = smooth/water, bright = rough/urban).
Map.addLayer(s1VV, {min: -25, max: 0}, 'Sentinel-1 VV', false);

// Display the VH composite (sensitive to vegetation structure).
Map.addLayer(s1VH, {min: -30, max: -5}, 'Sentinel-1 VH', false);

// ---------------------------------------------------------------------------
// SECTION 3 — SENTINEL-2 OPTICAL (cloud-free composite)
// ---------------------------------------------------------------------------

// Load the harmonised Sentinel-2 Level-2A surface-reflectance archive.
var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
  .filterBounds(aoi)                          // keep tiles touching the AOI
  .filterDate(startDate, endDate);            // keep tiles inside the time window

// Load Google's Cloud Score+ collection (state-of-the-art S2 cloud scoring).
var csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED');

// Choose the Cloud Score+ quality band: 'cs' = per-pixel clear-sky score 0–1.
var QA_BAND = 'cs';

// Pixels with a clear-sky score of at least this value are kept (0.6 is a good default).
var CLEAR_THRESHOLD = 0.6;

// Attach the matching Cloud Score+ image to every S2 tile, then mask cloudy pixels.
var s2Masked = s2
  .linkCollection(csPlus, [QA_BAND])                          // join the 'cs' band onto each S2 image by ID
  .map(function(img) {                                        // for every joined image...
    return img.updateMask(img.select(QA_BAND)                 // ...select its clear-sky score band...
                             .gte(CLEAR_THRESHOLD));          // ...and mask out pixels below the threshold
  });

// Print how many Sentinel-2 tiles matched (check the Console tab).
print('Sentinel-2 tiles found:', s2Masked.size());

// Build the per-pixel MEDIAN composite (clouds already removed), scale to reflectance.
var s2Composite = s2Masked
  .median()                                   // per-pixel median = clean, cloud-free mosaic
  .divide(10000)                              // convert DN (0–10000) to reflectance (0–1)
  .clip(aoi);                                 // clip to the study area

// Display a natural-colour preview of the composite (Red, Green, Blue bands).
Map.addLayer(s2Composite, {bands: ['B4','B3','B2'], min: 0, max: 0.3}, 'Sentinel-2 RGB', false);

// ---------------------------------------------------------------------------
// SECTION 4 — SPECTRAL INDICES (NDVI, NDWI, MNDWI)
// ---------------------------------------------------------------------------

// NDVI = (NIR - Red) / (NIR + Red) = (B8 - B4) / (B8 + B4) — vegetation vigour.
var ndvi = s2Composite.normalizedDifference(['B8', 'B4'])    // built-in normalised difference
  .rename('NDVI');                                           // clean band name

// NDWI = (Green - NIR) / (Green + NIR) = (B3 - B8) / (B3 + B8) — open-water index.
var ndwi = s2Composite.normalizedDifference(['B3', 'B8'])    // McFeeters (1996) formulation
  .rename('NDWI');                                           // clean band name

// MNDWI = (Green - SWIR1) / (Green + SWIR1) = (B3 - B11) / (B3 + B11) — better water/built-up separation.
var mndwi = s2Composite.normalizedDifference(['B3', 'B11'])  // Xu (2006) modified water index
  .rename('MNDWI');                                          // clean band name

// Display NDVI with a brown-to-green palette (initially hidden).
Map.addLayer(ndvi, {min: -0.2, max: 0.9, palette: ['a52a2a','ffff66','006400']}, 'NDVI', false);

// Display NDWI with a white-to-blue palette (initially hidden).
Map.addLayer(ndwi, {min: -0.5, max: 0.5, palette: ['ffffff','0000ff']}, 'NDWI', false);

// Display MNDWI with a white-to-dark-blue palette (initially hidden).
Map.addLayer(mndwi, {min: -0.5, max: 0.5, palette: ['ffffff','00008b']}, 'MNDWI', false);

// ---------------------------------------------------------------------------
// SECTION 5 — RAINFALL (CHIRPS mean annual precipitation)
// ---------------------------------------------------------------------------

// Load the CHIRPS daily precipitation archive (0.05 degree, ~5.5 km).
var chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY');

// Build a server-side list of years for the rainfall climatology.
var years = ee.List.sequence(rainStartYear, rainEndYear);

// For every year: sum all daily rainfall images into one annual-total image.
var annualTotals = ee.ImageCollection(                       // wrap the mapped list back into a collection
  years.map(function(y) {                                    // loop over each year server-side
    return chirps                                            // start from the daily archive
      .filter(ee.Filter.calendarRange(y, y, 'year'))         // keep only days inside year y
      .sum()                                                 // add them up = annual total (mm)
      .set('year', y);                                       // store the year as a property
  })
);

// Average the annual totals over all years = mean annual rainfall (mm/yr).
var rainfall = annualTotals
  .mean()                                     // per-pixel mean of the annual sums
  .clip(aoi)                                  // clip to the study area
  .rename('rain_annual_mm');                  // clean band name

// Display mean annual rainfall with a dry-to-wet palette (initially hidden).
Map.addLayer(rainfall, {min: 200, max: 1400, palette: ['ffffcc','41b6c4','0c2c84']}, 'Rainfall', false);

// ---------------------------------------------------------------------------
// SECTION 6 — EXPORT EVERY LAYER AS GeoTIFF TO GOOGLE DRIVE
// ---------------------------------------------------------------------------

// Small helper so each export is one readable call instead of ten repeated blocks.
var exportLayer = function(image, name) {     // takes the image and a file name
  Export.image.toDrive({                      // queue a Drive export task
    image: image.toFloat(),                   // cast to Float32 so all layers share one data type
    description: name,                        // task name shown in the Tasks tab
    fileNamePrefix: name,                     // output file name (name.tif)
    folder: driveFolder,                      // Drive folder that receives the file
    region: aoi,                              // export footprint = study area
    scale: outScale,                          // pixel size in metres (30 m common grid)
    crs: outCRS,                              // coordinate system (UTM 43N)
    maxPixels: 1e13,                          // raise the pixel cap for large areas
    fileFormat: 'GeoTIFF'                     // output format
  });
};

// Queue the DEM export.
exportLayer(dem,        'swat_dem');

// Queue the slope export.
exportLayer(slope,      'swat_slope');

// Queue the aspect export.
exportLayer(aspect,     'swat_aspect');

// Queue the Sentinel-1 VV composite export.
exportLayer(s1VV,       'swat_s1_vv');

// Queue the Sentinel-1 VH composite export.
exportLayer(s1VH,       'swat_s1_vh');

// Queue the Sentinel-2 core-band composite export (10/20 m bands used downstream).
exportLayer(s2Composite.select(['B2','B3','B4','B8','B11','B12']), 'swat_s2_composite');

// Queue the NDVI export.
exportLayer(ndvi,       'swat_ndvi');

// Queue the NDWI export.
exportLayer(ndwi,       'swat_ndwi');

// Queue the MNDWI export.
exportLayer(mndwi,      'swat_mndwi');

// Queue the rainfall export (native ~5.5 km data resampled onto the 30 m grid —
// state this resampling explicitly in your paper's methods section).
exportLayer(rainfall,   'swat_rainfall');

// Final console message: everything is queued; go press RUN on the Tasks tab.
print('All 10 export tasks queued. Open the Tasks tab (top right) and click RUN on each.');

timestep = '1 M'

# Set your own Manning values (default --> (min = 0.025; max = 0.055)):
minManning = 0.025
maxManning = 0.055
# 3 options are availables for Manning coefficient calculation, choose one of them (see manual)
ManningOption = '1'
# for option 3 you need a default manning value:
ManningValue = 0

# do you want to calculate the runoff coefficient based on an imperviousness map?
#(if yes, make sure your imperviousness map is saved in the folder basic_maps as "imperviousness_start")
imp_map = 'N'
# if not, what is the value of imperviousness in your urban cells?
imp_value = 0.5

# parameters for depression capacity calculation (see manual for more details)
b = 9.5
Sdu = 0.5

# would you like to use the default thresholds for the calculation of velocity?
# (defaults: min = 0.001, max = 3.0)
defaultVelocity = 'Y'
# if not, set your own:
minVelocity = 0
maxVelocity = 0

# parameters for flow direction calculation (in order to remove false depressions)
Core_Depth = 99999999999999999999999999999999999999999999999999999999999999999
Core_Volume = 9999999999999999999999999999999999999999999999999999999999999999999
Core_Area = 9999999999999999999999999999999999999999999999999999999999999999999999
Catchment_Precipitation = 9999999999999999999999999999999999999999999999999999999

# threshold to derive the stream network form the flow accumulation
Stream_threshold = 200

# would you like to set a min threshold for the slope map calculation?
Slope_min = 'Y'
# if so, what is the value of your threshold?
Slope_threshold = 0.0008
# would you like to create a map that visualize the region with slope< slope_threshold?
Slope_threshold_map = 'Y'

# return period for hydraulic radius calculation (choices: t2, t10, t100)
return_period = 't2'

# threshold to derive the sub-catchments form the flow accumulation
Subcatchment_threshold = 3000

#parameters for Initial Soil Moisture calculation
Smin = 0.8
Smax = 1.0



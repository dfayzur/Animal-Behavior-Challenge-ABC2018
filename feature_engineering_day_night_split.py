# import necessary modules
import datetime as dt
import pyproj
import os
from pathlib import Path
from sklearn import decomposition
geod = pyproj.Geod(ellps='WGS84')

import pandas as pd, numpy as np

from sklearn.preprocessing import MinMaxScaler

def to_datetime(string):
    return dt.datetime.strptime(string, '%Y-%m-%d %H:%M:%S')

def calculate_distance(long1, lat1, long2, lat2):
    if lat1 == lat2 and long1 == long2:
        return 0
    if False in np.isfinite([long1, long2, lat1, lat2]):
        return np.nan
    if lat1 < -90 or lat1 > 90 or lat2 < -90 or lat2 > 90:
        #raise ValueError('The range of latitudes seems to be invalid.')
        return np.nan
    if long1 < -180 or long1 > 180 or long2 < -180 or long2 > 180:
        return np.nan
        #raise ValueError('The range of longitudes seems to be invalid.')
    angle1,angle2,distance = geod.inv(long1, lat1, long2, lat2)
    return distance

def calculate_velocity(distance, timedelta):
    if timedelta == 0: return np.nan
    return distance / timedelta

def calculate_acceleration(velocity, velocity_next_position, timedelta):
    delta_v = velocity_next_position - velocity
    if timedelta == 0: return np.nan
    return delta_v / timedelta

headers_trajectory = ['longitude','latitude','azimus','elevation','day/night','esec','JST','days']#['lat', 'long', 'null', 'altitude', 'timestamp_float', 'date', 'time']

headers_trajectory_pca = ['long_next_position','lat_next_position','azimus_next_position','elevation_next_position','longitude','latitude','azimus','elevation', 'velocity_next_position', 'velocity']

def load_trajectory_df(full_filename, label):

    df = pd.read_csv(full_filename, header=None, names=headers_trajectory)

    df['long_next_position'] = df['longitude'].shift(-1)
    df['lat_next_position'] = df['latitude'].shift(-1)
    df['azimus_next_position'] = df['azimus'].shift(-1)
    df['elevation_next_position'] = df['elevation'].shift(-1)


    df['esec_next_position'] = df['esec'].shift(-1)
    df['timedelta'] = df.apply(lambda z: z.esec_next_position - z.esec, axis=1) #df.esec.values #

    df['long_delta'] = df['long_next_position'] - df['longitude']

    df['latitude_delta'] = df['lat_next_position'] - df['latitude']

    R = 6371

    x_1 = (R + df['elevation'] * np.pi / 180) * np.cos(df['latitude'] * np.pi / 180) * np.sin(df['longitude'] * np.pi / 180)
    y_1 = (R + df['elevation'] * np.pi / 180) * np.sin(df['latitude'] * np.pi / 180)
    z_1 = (R + df['elevation'] * np.pi / 180) * np.cos(df['latitude'] * np.pi / 180) * np.cos(df['longitude'] * np.pi / 180)

    x_2 = (R + df['elevation_next_position'] * np.pi / 180) * np.cos(df['lat_next_position'] * np.pi / 180) * np.sin(df['long_next_position'] * np.pi / 180)
    y_2 = (R + df['elevation_next_position'] * np.pi / 180) * np.sin(df['lat_next_position'] * np.pi / 180)
    z_2 = (R + df['elevation_next_position'] * np.pi / 180) * np.cos(df['lat_next_position'] * np.pi / 180) * np.cos(df['long_next_position'] * np.pi / 180)

    dist_= np.sqrt((x_2-x_1)**2 + (y_2-y_1)**2 + (z_2-z_1)**2) * 1000


    df['distance'] = dist_



    df['velocity'] = df.apply(lambda z: calculate_velocity(z.distance, z.timedelta), axis=1)
    df['velocity_next_position'] = df['velocity'].shift(-1)
    df['velocity_delta'] = df['velocity_next_position'] - df['velocity']
    df['acceleration'] = df.apply(lambda z: calculate_acceleration(z.velocity, z.velocity_next_position, z.timedelta),
                                  axis=1)

    pca = decomposition.PCA(n_components=2)

    X = df[headers_trajectory_pca]
    X = X.dropna(how='any', axis=0)

    X = X.values
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    X = X.transpose()

    X = X[:, :-1]
    pca.fit(X)

    X = pca.transform(X)

    df = df.drop(['long_next_position', 'lat_next_position'], axis=1)
    df = df.drop(['velocity_next_position'], axis=1)


    df['azimus_delta'] = df['azimus_next_position'] - df['azimus']

    df['elevation_delta'] = df['elevation_next_position'] - df['elevation']

    df = df.drop(['azimus_next_position', 'elevation_next_position'], axis=1)



    df['bird_id'] = str(full_filename).split('\\')[2].split('.')[0]

    df['labels'] = ''
    df_ = calculate_agg_features(df, label, X)

    return df_

def calculate_agg_middle_features(df, velocity_median, velocity_mean, velocity_05, velocity_10, velocity_15, velocity_25, velocity_75, velocity_85, velocity_90, velocity_95, velocity_99):
    if df.shape[0]>0:
        # This method calculates the aggregated feature and
        # saves them in the original df as well as an metadata df.
        v_ave = np.nanmean(df['velocity'].values)
        v_min = np.nanmin(df['velocity'].values)

        v_max = np.nanmax(df['velocity'].values)
        a_ave = np.nanmean(df['acceleration'].values)

        a_min = np.nanmin(df['acceleration'].values)
        a_max = np.nanmax(df['acceleration'].values)

        d_ave = np.nanmean(df['distance'].values)

        d_min = np.nanmin(df['distance'].values)
        d_max = np.nanmax(df['distance'].values)

        e_ave = np.nanmean(df['elevation'].values)

        e_min = np.nanmin(df['elevation'].values)
        e_max = np.nanmax(df['elevation'].values)

        lon_ave = np.nanmean(df['longitude'].values)

        lon_min = np.nanmin(df['longitude'].values)
        lon_max = np.nanmax(df['longitude'].values)

        lat_ave = np.nanmean(df['latitude'].values)

        lat_min = np.nanmin(df['latitude'].values)
        lat_max = np.nanmax(df['latitude'].values)

        az_ave = np.nanmean(df['azimus'].values)

        az_min = np.nanmin(df['azimus'].values)
        az_max = np.nanmax(df['azimus'].values)

        long_delta_ave = np.nanmean(df['long_delta'].values)

        long_delta_min = np.nanmin(df['long_delta'].values)
        long_delta_max = np.nanmax(df['long_delta'].values)

        latitude_delta_ave = np.nanmean(df['latitude_delta'].values)

        latitude_delta_min = np.nanmin(df['latitude_delta'].values)
        latitude_delta_max = np.nanmax(df['latitude_delta'].values)

        velocity_delta_ave = np.nanmean(df['velocity'].values)

        velocity_delta_min = np.nanmin(df['velocity'].values)
        velocity_delta_max = np.nanmax(df['velocity'].values)

        azimus_delta_ave = np.nanmean(df['azimus'].values)

        azimus_delta_min = np.nanmin(df['azimus'].values)
        azimus_delta_max = np.nanmax(df['azimus'].values)

        elevation_delta_ave = np.nanmean(df['elevation'].values)

        elevation_delta_min = np.nanmin(df['elevation'].values)
        elevation_delta_max = np.nanmax(df['elevation'].values)

        velocity_median_count = np.sum(df['velocity'] > velocity_median)
        velocity_mean_count = np.sum(df['velocity'] > velocity_mean)


        velocity_05_count = np.sum(df['velocity'] > velocity_05)
        velocity_10_count = np.sum(df['velocity'] > velocity_10)
        velocity_15_count = np.sum(df['velocity'] > velocity_15)

        velocity_25_count = np.sum(df['velocity'] > velocity_25)
        velocity_75_count = np.sum(df['velocity'] > velocity_75)
        velocity_85_count = np.sum(df['velocity'] > velocity_85)

        velocity_90_count = np.sum(df['velocity'] > velocity_90)
        velocity_95_count = np.sum(df['velocity'] > velocity_95)
        velocity_99_count = np.sum(df['velocity'] > velocity_99)



        middle_list = list(df['distance'].quantile([0, .05, .1, .2, .25, .3, .4, .5, .6, .7, .75, .8, .9, .95, 1])) + \
        list(df['velocity'].quantile([0, .05, .1, .2, .25, .3, .4, .5, .6, .7, .75, .8, .9, .95, 1])) + \
        list(df['acceleration'].quantile([0, .05, .1, .2, .25, .3, .4, .5, .6, .7, .75, .8, .9, .95, 1])) + \
        list(df['elevation'].quantile([0, .05, .1, .2, .25, .3, .4, .5, .6, .7, .75, .8, .9, .95, 1])) + \
        list(df['longitude'].quantile([0, .05, .1, .2, .25, .3, .4, .5, .6, .7, .75, .8, .9, .95, 1])) + \
        list(df['latitude'].quantile([0, .05, .1, .2, .25, .3, .4, .5, .6, .7, .75, .8, .9, .95, 1])) + \
        list(df['azimus'].quantile([0, .05, .1, .2, .25, .3, .4, .5, .6, .7, .75, .8, .9, .95, 1])) + \
        list(df['long_delta'].quantile([0, .05, .1, .2, .25, .3, .4, .5, .6, .7, .75, .8, .9, .95, 1])) + \
        list(df['latitude_delta'].quantile([0, .05, .1, .2, .25, .3, .4, .5, .6, .7, .75, .8, .9, .95, 1])) + \
        list(df['velocity_delta'].quantile([0, .05, .1, .2, .25, .3, .4, .5, .6, .7, .75, .8, .9, .95, 1])) + \
        list(df['azimus_delta'].quantile([0, .05, .1, .2, .25, .3, .4, .5, .6, .7, .75, .8, .9, .95, 1])) + \
        list(df['elevation_delta'].quantile([0, .05, .1, .2, .25, .3, .4, .5, .6, .7, .75, .8, .9, .95, 1])) + \
        [d_ave, d_min, d_max] + \
        [v_ave, v_min, v_max] + \
        [a_ave, a_min, a_max] + \
        [e_ave, e_min, e_max] + \
        [lon_ave, lon_min, lon_max] + \
        [lat_ave, lat_min, lat_max] + \
        [az_ave, az_min, az_max] + \
        [long_delta_ave, long_delta_min, long_delta_max] + \
        [latitude_delta_ave, latitude_delta_min, latitude_delta_max] + \
        [velocity_delta_ave, velocity_delta_min, velocity_delta_max] + \
        [azimus_delta_ave, azimus_delta_min, azimus_delta_max] + \
        [elevation_delta_ave, elevation_delta_min, elevation_delta_max] + \
        [velocity_median_count, velocity_mean_count, velocity_05_count, velocity_10_count, velocity_15_count, velocity_25_count, velocity_75_count, velocity_85_count, velocity_90_count, velocity_95_count, velocity_99_count]
    else:
        middle_list = [-1.0] * 227

    return middle_list

def calculate_agg_features(df, label, X):
    df_day = df.loc[df['day/night'] == 1].copy(deep=True)
    df_night = df.loc[df['day/night'] == 0].copy(deep=True)
    middle_list_day = calculate_agg_middle_features(df_day, velocity_median=1.925621, velocity_mean=4.153017, velocity_05=0.159845, velocity_10=0.229731, velocity_15=0.301470, velocity_25=0.447886, velocity_75=7.899708, velocity_85=9.550042, velocity_90=10.474088, velocity_95=11.709188, velocity_99=14.049555)
    middle_list_night = calculate_agg_middle_features(df_night, velocity_median=0.576698, velocity_mean=2.465827, velocity_05=0.142663, velocity_10=0.171406, velocity_15=0.218119, velocity_25=0.305110, velocity_75=3.133845, velocity_85=7.252443,velocity_90=8.718294, velocity_95=10.506525, velocity_99=13.320440)
    df_ = np.array([int(df.bird_id.values[0])] +
           middle_list_day +
           middle_list_night +
           X.flatten().tolist() +
           df.latitude.values[:5].tolist()+
           df.longitude.values[:5].tolist()+
           [abs(df.latitude.max() - df.latitude.min()), abs(df.longitude.max() - df.longitude.min())] +
           [label])

    return df_

csv_files = sorted(Path(os.path.join('.','data','train')).glob('*.csv'))
train_label = np.genfromtxt(os.path.join('.','data','train_labels.csv'), delimiter=',') #np.genfromtxt('train_labels.csv', delimiter=',')

col_names =  ['bird_id',
    'q_00_distance', 'q_05_distance', 'q_10_distance', 'q_20_distance', 'q_25_distance', 'q_30_distance', 'q_40_distance', 'q_50_distance', 'q_60_distance', 'q_70_distance', 'q_75_distance', 'q_80_distance', 'q_90_distance', 'q_95_distance', 'q_100_distance',
    'q_00_velocity', 'q_05_velocity', 'q_10_velocity', 'q_20_velocity', 'q_25_velocity', 'q_30_velocity', 'q_40_velocity', 'q_50_velocity', 'q_60_velocity', 'q_70_velocity', 'q_75_velocity', 'q_80_velocity', 'q_90_velocity', 'q_95_velocity', 'q_100_velocity',
    'q_00_acceleration', 'q_05_acceleration', 'q_10_acceleration', 'q_20_acceleration', 'q_25_acceleration', 'q_30_acceleration', 'q_40_acceleration', 'q_50_acceleration', 'q_60_acceleration', 'q_70_acceleration', 'q_75_acceleration', 'q_80_acceleration', 'q_90_acceleration', 'q_95_acceleration', 'q_100_acceleration',
    'q_00_elevation', 'q_05_elevation', 'q_10_elevation', 'q_20_elevation', 'q_25_elevation', 'q_30_elevation', 'q_40_elevation', 'q_50_elevation', 'q_60_elevation', 'q_70_elevation', 'q_75_elevation', 'q_80_elevation', 'q_90_elevation', 'q_95_elevation', 'q_100_elevation',
    'q_00_long', 'q_05_long', 'q_10_long', 'q_20_long', 'q_25_long', 'q_30_long', 'q_40_long', 'q_50_long', 'q_60_long', 'q_70_long', 'q_75_long', 'q_80_long', 'q_90_long', 'q_95_long', 'q_100_long',
    'q_00_lat', 'q_05_lat', 'q_10_lat', 'q_20_lat', 'q_25_lat', 'q_30_lat', 'q_40_lat', 'q_50_lat', 'q_60_lat', 'q_70_lat', 'q_75_lat', 'q_80_lat', 'q_90_lat', 'q_95_lat', 'q_100_lat',
    'q_00_az', 'q_05_az', 'q_10_az', 'q_20_az', 'q_25_az', 'q_30_az', 'q_40_az', 'q_50_az', 'q_60_az', 'q_70_az', 'q_75_az', 'q_80_az', 'q_90_az', 'q_95_az', 'q_100_az',
    'q_00_long_delta', 'q_05_long_delta', 'q_10_long_delta', 'q_20_long_delta', 'q_25_long_delta', 'q_30_long_delta', 'q_40_long_delta', 'q_50_long_delta', 'q_60_long_delta', 'q_70_long_delta', 'q_75_long_delta', 'q_80_long_delta', 'q_90_long_delta', 'q_95_long_delta', 'q_100_long_delta',
    'q_00_lat_delta', 'q_05_lat_delta', 'q_10_lat_delta', 'q_20_lat_delta', 'q_25_lat_delta', 'q_30_lat_delta', 'q_40_lat_delta', 'q_50_lat_delta', 'q_60_lat_delta', 'q_70_lat_delta', 'q_75_lat_delta', 'q_80_lat_delta', 'q_90_lat_delta', 'q_95_lat_delta', 'q_100_lat_delta',
    'q_00_vel_delta', 'q_05_vel_delta', 'q_10_vel_delta', 'q_20_vel_delta', 'q_25_vel_delta', 'q_30_vel_delta', 'q_40_vel_delta', 'q_50_vel_delta', 'q_60_vel_delta', 'q_70_vel_delta', 'q_75_vel_delta', 'q_80_vel_delta', 'q_90_vel_delta', 'q_95_vel_delta', 'q_100_vel_delta',
    'q_00_az_delta', 'q_05_az_delta', 'q_10_az_delta', 'q_20_az_delta', 'q_25_az_delta', 'q_30_az_delta', 'q_40_az_delta', 'q_50_az_delta', 'q_60_az_delta', 'q_70_az_delta', 'q_75_az_delta', 'q_80_az_delta', 'q_90_az_delta', 'q_95_az_delta', 'q_100_az_delta',
    'q_00_elev_delta', 'q_05_elev_delta', 'q_10_elev_delta', 'q_20_elev_delta', 'q_25_elev_delta', 'q_30_elev_delta', 'q_40_elev_delta', 'q_50_elev_delta', 'q_60_elev_delta', 'q_70_elev_delta', 'q_75_elev_delta', 'q_80_elev_delta', 'q_90_elev_delta', 'q_95_elev_delta', 'q_100_elev_delta',
    'mean_distance', 'min_distance', 'max_distance',
    'mean_velocity', 'min_velocity', 'max_velocity',
    'mean_acceleration', 'min_acceleration', 'max_acceleration',
    'mean_elevation', 'min_elevation', 'max_elevation',
    'mean_long', 'min_long', 'max_long',
    'mean_lat', 'min_lat', 'max_lat',
    'mean_az', 'min_az', 'max_az',
    'mean_long_delta', 'min_long_delta', 'max_long_delta',
    'mean_latitude_delta', 'min_latitude_delta', 'max_latitude_delta',
    'mean_velocity_delta', 'min_velocity_delta', 'max_velocity_delta',
    'mean_azimus_delta', 'min_azimus_delta', 'max_azimus_delta',
    'mean_elevation_delta', 'min_elevation_delta', 'max_elevation_delta',
    'velocity_median_count', 'velocity_mean_count', 'velocity_05_count', 'velocity_10_count', 'velocity_15_count', 'velocity_25_count', 'velocity_75_count', 'velocity_85_count', 'velocity_90_count', 'velocity_95_count', 'velocity_99_count',
    'label']

day_col_names = ["day_" + name for name in col_names[1:-1]]
night_col_names = ["night_" + name for name in col_names[1:-1]]
day_night_col_names = ['bird_id'] + day_col_names + night_col_names + \
                      ['PCA1', 'PCA2', 'PCA3', 'PCA4', 'PCA5', 'PCA6', 'PCA7', 'PCA8', 'PCA9', 'PCA10', 'PCA11',
                       'PCA12', 'PCA13', 'PCA14', 'PCA15', 'PCA16', 'PCA17', 'PCA18', 'PCA19', 'PCA20'] + \
                      ['lat1', 'lat2', 'lat3','lat4', 'lat5', 'long1', 'long2', 'long3','long4', 'long5'] +\
                      ['abs_lat', 'abs_long'] +\
                      ['label']

train_df_np = np.zeros(shape=(len(csv_files),len(day_night_col_names)))

for i in range(len(csv_files)):

    print("train: " + str(i))

    label_ = train_label[i]
    print("label_ : ", label_)
    train_df_np[i,:] = load_trajectory_df(csv_files[i], label_)



train_df  = pd.DataFrame(train_df_np,columns = day_night_col_names)
train_df.fillna(-1, inplace=True)
#train_df.to_csv("train_df_new_last.csv", index=False)
train_df.to_csv(os.path.join('.','data', 'train_df_day_night_split.csv'), index=False)

csv_files_test = sorted(Path(os.path.join('.','data','test')).glob('*.csv'))


test_df_np = np.zeros(shape=(len(csv_files_test),len(day_night_col_names)))

for i in range(len(csv_files_test)):
    print("test: " + str(i))

    label_ = -1
    test_df_np[i,:] = load_trajectory_df(csv_files_test[i], label_)


test_df = pd.DataFrame(test_df_np,columns = day_night_col_names)
test_df.fillna(-1, inplace=True)
#test_df.to_csv("test_df_new_last.csv", index=False)
test_df.to_csv(os.path.join('.','data', 'test_df_day_night_split.csv'), index=False)

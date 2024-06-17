import sqlite3

class SqliteDataProvider:

    def __init__(self, sqlite_path=None, lsqlite_path=None):
        self.sqlite_path = sqlite_path
        self.lsqlite_path = lsqlite_path
        # if self.sqlite_path:
        #     self.s_conn = sqlite3.connect(self.sqlite_path, check_same_thread=False)
        # if self.lsqlite_path:
        #     self.ls_conn = sqlite3.connect(self.lsqlite_path, check_same_thread=False)

    def _generic_select(self, connection, query, params=None):
        cur = connection.cursor()
        if params:
            cur.execute(query, params)
        else:
            cur.execute(query, )
        result = cur.fetchall()
        if len(result) == 0 or result is None:
            raise DataNotFoundError(f"No records returned for a query: {query}, params: {params}")
        return result

    def _select_from_sqlite(self, query, params=None):
        if not self.sqlite_path:
            raise PathUndefinedError("SQLITE path is not defined")
        with sqlite3.connect(self.sqlite_path) as con: # something here
            return self._generic_select(con, query, params)
        # return self._generic_select(self.s_conn, query, params)

    def _select_from_lsqlite(self, query, params=None):
        if not self.lsqlite_path:
            raise PathUndefinedError("LSQLITE path is not defined")
        with sqlite3.connect(self.lsqlite_path) as con:
            return self._generic_select(con, query, params)
        # return self._generic_select(self.ls_conn, query, params)

    def get_laser_data_table(self, laser_id):
        query = "SELECT laserDataTableName FROM laser_index WHERE laserId = ?"
        params = (laser_id,)
        return self._select_from_lsqlite(query, params)[0][0]

    def get_laser_metadata_table(self, laser_id):
        query = "SELECT laserMetaDataTableName FROM laser_index WHERE laserId = ?"
        params = (laser_id,)
        return self._select_from_lsqlite(query, params)[0][0]

    def get_blob_count(self, laser_id):
        laser_data_table = self.get_laser_data_table(laser_id)
        query = f"SELECT count(*) FROM {laser_data_table}"
        return self._select_from_lsqlite(query)[0][0]

    def get_laser_blob(self, laser_id, blob_id):
        laser_data_table = self.get_laser_data_table(laser_id)
        query = f"SELECT data FROM {laser_data_table} WHERE idx = ?"
        params = (blob_id,)
        return self._select_from_lsqlite(query, params)[0][0]

    def get_laser_format(self, laser_id):
        laser_metadata_table = self.get_laser_metadata_table(laser_id)
        try:
            query = f"SELECT data FROM {laser_metadata_table} AS mt " \
                    f"INNER JOIN laser_metadata_types AS ts ON ts.type = mt.type " \
                    f"WHERE description = ?"
            params = ("LASER FORMAT",)
            return self._select_from_lsqlite(query, params)[0][0].decode("utf-8")
        except DataNotFoundError:
            query = f"SELECT data FROM {laser_metadata_table} WHERE type = ?"
            params = (1,)
            return self._select_from_lsqlite(query, params)[0][0].decode("utf-8")

    def get_time_index(self, laser_id):
        laser_metadata_table = self.get_laser_metadata_table(laser_id)
        try:
            query = f"SELECT data FROM {laser_metadata_table} AS mt " \
                    f"INNER JOIN laser_metadata_types AS ts ON ts.type = mt.type " \
                    f"WHERE description = ?"
            params = ("LASER TIME INDEX",)
            return self._select_from_lsqlite(query, params)[0][0]
        except DataNotFoundError:
            query = f"SELECT data FROM {laser_metadata_table} WHERE type = ?"
            params = (2,)
            return self._select_from_lsqlite(query, params)[0][0]

    def get_laser_calibration(self, laser_id):
        query = "SELECT ttvalue FROM calibrations WHERE ttkey = ? AND type = ?"
        params = (laser_id, "position")
        calibration = self._select_from_lsqlite(query, params)[0][0]
        if type(calibration) is str:
            return calibration
        return calibration.decode("utf8")

    def get_camera_calibration(self, camera_id):
        query = "SELECT ttvalue FROM calibrations WHERE ttkey = ? AND type = ?"
        params = (camera_id, "position")
        calibration = self._select_from_sqlite(query, params)[0][0]
        if type(calibration) is str:
            return calibration
        return calibration.decode("utf8")

    def get_trajectory(self):
        query = "SELECT ttvalue FROM positions WHERE ttkey = ?"
        params = ("apx3",)
        return self._select_from_sqlite(query, params)[0][0]

    def get_gps(self):
        query = "SELECT ttvalue FROM sessionfiles WHERE ttkey = ?"
        params = ("gps",)
        return self._select_from_sqlite(query, params)[0][0]

    def get_tiles(self, frame_table_name, frame_id):
        query = f"SELECT ttvalue from {frame_table_name} where idx=? order by part"
        params = (frame_id,)
        return [record[0] for record in self._select_from_sqlite(query, params)]

    def get_camera_by_type(self, camera_type):
        query = "SELECT ttvalue FROM cameras WHERE type=?"
        params = (camera_type,)
        return self._select_from_sqlite(query, params)[0][0]

    def get_session_metadata(self, metadata_key):
        query = "SELECT ttvalue FROM metadata WHERE ttkey=?"
        params = (metadata_key,)
        return self._select_from_sqlite(query, params)[0][0]

    def get_frame_positions(self, table_name):
        query = f"SELECT frametime, longitude, latitude, altitude, yaw, pitch, roll " \
                f"FROM {table_name} ORDER BY idx"
        return [
            {"frame_time": record[0],
             "longitude": record[1],
             "latitude": record[2],
             "altitude": record[3],
             "yaw": record[4],
             "pitch": record[5],
             "roll": record[6]}
            for record in self._select_from_sqlite(query)
        ]

    def get_pano_time(self, table_name):
        query = f"SELECT frametime " \
                f"FROM {table_name} ORDER BY idx"
        return [
            record[0]
            for record in self._select_from_sqlite(query)
        ]

class DataNotFoundError(Exception):
    pass


class PathUndefinedError(Exception):
    pass

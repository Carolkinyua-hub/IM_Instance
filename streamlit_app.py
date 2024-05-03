import sqlite3

class StreamlitApp:
    def __init__(self, db_name='streamlit_db.sqlite'):
        self.db_name = db_name
        self.connection = sqlite3.connect(db_name)
        self.cursor = self.connection.cursor()
        self.create_table()

    def create_table(self):
        create_users_table = """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            username TEXT,
            email TEXT UNIQUE,
            password TEXT
        );
        """
        self.cursor.execute(create_users_table)
        self.connection.commit()

    def create_user(self, username, email, password):
        try:
            insert_user_query = """
            INSERT INTO users (username, email, password)
            VALUES (?, ?, ?);
            """
            self.cursor.execute(insert_user_query, (username, email, password))
            self.connection.commit()
            return True
        except sqlite3.IntegrityError:
            return False  # User with this email already exists

    def login(self, email, password):
        select_user_query = """
        SELECT * FROM users
        WHERE email = ? AND password = ?;
        """
        self.cursor.execute(select_user_query, (email, password))
        user = self.cursor.fetchone()
        if user:
            return user  # Returning the user row
        else:
            return None  # Invalid email or password

    def __del__(self):
        self.connection.close()

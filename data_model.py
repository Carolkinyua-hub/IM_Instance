class User:
    def __init__(self, username, email, password):
        self.username = username
        self.email = email
        self.password = password

class Post:
    def __init__(self, title, content, author):
        self.title = title
        self.content = content
        self.author = author
        self.likes = 0
        self.comments = []

    def add_comment(self, comment):
        self.comments.append(comment)

class StreamlitApp:
    def __init__(self):
        self.users = {}
        self.posts = []

    def create_user(self, username, email, password):
        if email in self.users:
            return None  # User with this email already exists
        user = User(username, email, password)
        self.users[email] = user
        return user

    def create_post(self, title, content, author_email):
        if author_email not in self.users:
            return None  # Author does not exist
        author = self.users[author_email]
        post = Post(title, content, author)
        self.posts.append(post)
        return post

    def like_post(self, post):
        post.likes += 1

    def comment_on_post(self, post, comment):
        post.add_comment(comment)

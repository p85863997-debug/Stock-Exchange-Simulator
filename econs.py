"""
Advanced Multi-Feature Chat Application
A production-ready messaging platform combining features from WhatsApp, Snapchat, and Telegram
"""

import streamlit as st
import asyncio
import json
import hashlib
import jwt
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import base64
import io
import re
import uuid
import time
from dataclasses import dataclass, asdict
from enum import Enum
import pickle
import os
from pathlib import Path

# Configure Streamlit page
st.set_page_config(
    page_title="ChatFusion Pro",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern UI
st.markdown("""
<style>
    /* Modern color scheme */
    :root {
        --primary-color: #5865F2;
        --secondary-color: #7289DA;
        --success-color: #43B581;
        --danger-color: #F04747;
        --warning-color: #FAA61A;
        --dark-bg: #2C2F33;
        --darker-bg: #23272A;
        --light-text: #FFFFFF;
        --muted-text: #D9D9D9;
        --dark-text: #1a1a1a;
        --message-bg: #40444B;
        --online-status: #43B581;
        --away-status: #FAA61A;
        --offline-status: #747F8D;
    }

    /* Global text color */
    body, .stApp {
        color: var(--light-text);
    }

    /* Chat interface */
    .chat-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        color: var(--dark-text); /* ensure visible text */
    }

    .message-bubble {
        max-width: 70%;
        padding: 12px 16px;
        margin: 8px 0;
        border-radius: 18px;
        word-wrap: break-word;
        animation: slideIn 0.3s ease;
    }

    .message-sent {
        background: linear-gradient(135deg, #5865F2, #7289DA);
        color: var(--light-text) !important;
        margin-left: auto;
        margin-right: 10px;
        border-bottom-right-radius: 4px;
    }

    .message-received {
        background: #E3E5E8;
        color: var(--dark-text) !important;
        margin-right: auto;
        margin-left: 10px;
        border-bottom-left-radius: 4px;
    }

    /* Group chat header */
    .group-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: var(--light-text) !important;
        padding: 15px;
        border-radius: 15px 15px 0 0;
        margin: -20px -20px 20px -20px;
    }

    /* Sidebar */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        color: var(--dark-text) !important;
    }

    /* Buttons */
    .custom-button {
        background: linear-gradient(135deg, #5865F2, #7289DA);
        color: var(--light-text) !important;
        border: none;
        padding: 10px 20px;
        border-radius: 25px;
        font-weight: 600;
        cursor: pointer;
        transition: all 0.3s;
    }

    .custom-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 20px rgba(88, 101, 242, 0.3);
    }

    /* Captions & muted */
    .caption, small, .muted-text {
        color: var(--muted-text) !important;
    }

    /* Reaction chips */
    .reaction-chip {
        background: rgba(88, 101, 242, 0.1);
        border: 1px solid rgba(88, 101, 242, 0.3);
        border-radius: 15px;
        padding: 2px 8px;
        font-size: 12px;
        cursor: pointer;
        transition: all 0.2s;
        color: var(--dark-text) !important; /* readable on light bg */
    }
    .reaction-chip:hover {
        background: rgba(88, 101, 242, 0.2);
        transform: scale(1.1);
    }

    /* Story items */
    .story-item small {
        color: var(--dark-text) !important; /* readable below avatar */
    }

    /* Notification badge */
    .notification-badge {
        background: var(--danger-color);
        color: var(--light-text) !important;
        border-radius: 50%;
        padding: 2px 6px;
        font-size: 12px;
        font-weight: bold;
        margin-left: auto;
    }
</style>
""", unsafe_allow_html=True)


# Database Models and Enums
class UserStatus(Enum):
    ONLINE = "online"
    AWAY = "away"
    OFFLINE = "offline"
    BUSY = "busy"

class MessageType(Enum):
    TEXT = "text"
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    FILE = "file"
    VOICE = "voice"
    LOCATION = "location"
    CONTACT = "contact"
    POLL = "poll"

class MessageStatus(Enum):
    SENT = "sent"
    DELIVERED = "delivered"
    READ = "read"

@dataclass
class User:
    user_id: str
    username: str
    email: str
    phone: Optional[str]
    avatar: Optional[str]
    status_message: str
    online_status: UserStatus
    last_seen: datetime
    created_at: datetime
    is_verified: bool = False
    two_factor_enabled: bool = False

@dataclass
class Message:
    message_id: str
    sender_id: str
    recipient_id: str
    content: str
    message_type: MessageType
    status: MessageStatus
    timestamp: datetime
    edited_at: Optional[datetime] = None
    reply_to: Optional[str] = None
    reactions: Dict[str, List[str]] = None
    is_deleted: bool = False
    expires_at: Optional[datetime] = None

@dataclass
class Group:
    group_id: str
    name: str
    description: str
    avatar: Optional[str]
    creator_id: str
    admin_ids: List[str]
    member_ids: List[str]
    created_at: datetime
    settings: Dict[str, Any]

@dataclass
class Story:
    story_id: str
    user_id: str
    content: str
    media_url: Optional[str]
    created_at: datetime
    expires_at: datetime
    views: List[str]
    reactions: Dict[str, List[str]]
    is_highlight: bool = False

# Database Manager
class DatabaseManager:
    def __init__(self, db_path="chat_app.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                user_id TEXT PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                phone TEXT,
                avatar TEXT,
                status_message TEXT,
                online_status TEXT,
                last_seen TIMESTAMP,
                created_at TIMESTAMP,
                is_verified BOOLEAN DEFAULT FALSE,
                two_factor_enabled BOOLEAN DEFAULT FALSE,
                settings TEXT
            )
        """)
        
        # Messages table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                message_id TEXT PRIMARY KEY,
                sender_id TEXT NOT NULL,
                recipient_id TEXT,
                group_id TEXT,
                content TEXT,
                message_type TEXT,
                status TEXT,
                timestamp TIMESTAMP,
                edited_at TIMESTAMP,
                reply_to TEXT,
                reactions TEXT,
                is_deleted BOOLEAN DEFAULT FALSE,
                expires_at TIMESTAMP,
                FOREIGN KEY (sender_id) REFERENCES users(user_id)
            )
        """)
        
        # Groups table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS groups (
                group_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                avatar TEXT,
                creator_id TEXT NOT NULL,
                admin_ids TEXT,
                member_ids TEXT,
                created_at TIMESTAMP,
                settings TEXT,
                FOREIGN KEY (creator_id) REFERENCES users(user_id)
            )
        """)
        
        # Stories table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS stories (
                story_id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                content TEXT,
                media_url TEXT,
                created_at TIMESTAMP,
                expires_at TIMESTAMP,
                views TEXT,
                reactions TEXT,
                is_highlight BOOLEAN DEFAULT FALSE,
                FOREIGN KEY (user_id) REFERENCES users(user_id)
            )
        """)
        
        # Contacts table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS contacts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                contact_id TEXT NOT NULL,
                nickname TEXT,
                is_blocked BOOLEAN DEFAULT FALSE,
                is_favorite BOOLEAN DEFAULT FALSE,
                added_at TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(user_id),
                FOREIGN KEY (contact_id) REFERENCES users(user_id),
                UNIQUE(user_id, contact_id)
            )
        """)
        
        # Channels table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS channels (
                channel_id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                avatar TEXT,
                owner_id TEXT NOT NULL,
                moderator_ids TEXT,
                subscriber_ids TEXT,
                category TEXT,
                is_public BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP,
                settings TEXT,
                FOREIGN KEY (owner_id) REFERENCES users(user_id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def create_user(self, username: str, email: str, password: str) -> Optional[User]:
        """Create a new user account"""
        user_id = str(uuid.uuid4())
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                INSERT INTO users (
                    user_id, username, email, password_hash, 
                    status_message, online_status, last_seen, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                user_id, username, email, password_hash,
                "Hey there! I'm using ChatFusion", UserStatus.ONLINE.value,
                datetime.now(), datetime.now()
            ))
            conn.commit()
            
            user = User(
                user_id=user_id,
                username=username,
                email=email,
                phone=None,
                avatar=None,
                status_message="Hey there! I'm using ChatFusion",
                online_status=UserStatus.ONLINE,
                last_seen=datetime.now(),
                created_at=datetime.now()
            )
            return user
            
        except sqlite3.IntegrityError:
            return None
        finally:
            conn.close()
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user login"""
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT user_id, username, email, phone, avatar, status_message,
                   online_status, last_seen, created_at, is_verified, two_factor_enabled
            FROM users
            WHERE username = ? AND password_hash = ?
        """, (username, password_hash))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return User(
                user_id=row[0],
                username=row[1],
                email=row[2],
                phone=row[3],
                avatar=row[4],
                status_message=row[5],
                online_status=UserStatus(row[6]),
                last_seen=row[7],
                created_at=row[8],
                is_verified=row[9],
                two_factor_enabled=row[10]
            )
        return None
    
    def send_message(self, sender_id: str, recipient_id: str, content: str, 
                    message_type: MessageType = MessageType.TEXT) -> Message:
        """Send a message to a user or group"""
        message_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO messages (
                message_id, sender_id, recipient_id, content, 
                message_type, status, timestamp
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            message_id, sender_id, recipient_id, content,
            message_type.value, MessageStatus.SENT.value, timestamp
        ))
        
        conn.commit()
        conn.close()
        
        return Message(
            message_id=message_id,
            sender_id=sender_id,
            recipient_id=recipient_id,
            content=content,
            message_type=message_type,
            status=MessageStatus.SENT,
            timestamp=timestamp,
            reactions={}
        )
    
    def get_messages(self, user_id: str, contact_id: str, limit: int = 50) -> List[Message]:
        """Get messages between two users"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT message_id, sender_id, recipient_id, content, message_type,
                   status, timestamp, edited_at, reply_to, reactions, is_deleted
            FROM messages
            WHERE (sender_id = ? AND recipient_id = ?) 
               OR (sender_id = ? AND recipient_id = ?)
            ORDER BY timestamp DESC
            LIMIT ?
        """, (user_id, contact_id, contact_id, user_id, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        messages = []
        for row in rows:
            reactions = json.loads(row[9]) if row[9] else {}
            messages.append(Message(
                message_id=row[0],
                sender_id=row[1],
                recipient_id=row[2],
                content=row[3],
                message_type=MessageType(row[4]),
                status=MessageStatus(row[5]),
                timestamp=row[6],
                edited_at=row[7],
                reply_to=row[8],
                reactions=reactions,
                is_deleted=row[10]
            ))
        
        return messages[::-1]  # Reverse to get chronological order
    
    def create_group(self, name: str, description: str, creator_id: str, 
                    member_ids: List[str]) -> Group:
        """Create a new group chat"""
        group_id = str(uuid.uuid4())
        created_at = datetime.now()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO groups (
                group_id, name, description, creator_id, 
                admin_ids, member_ids, created_at, settings
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            group_id, name, description, creator_id,
            json.dumps([creator_id]), json.dumps(member_ids),
            created_at, json.dumps({})
        ))
        
        conn.commit()
        conn.close()
        
        return Group(
            group_id=group_id,
            name=name,
            description=description,
            avatar=None,
            creator_id=creator_id,
            admin_ids=[creator_id],
            member_ids=member_ids,
            created_at=created_at,
            settings={}
        )
    
    def get_user_contacts(self, user_id: str) -> List[Dict]:
        """Get user's contact list"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT c.contact_id, u.username, u.avatar, u.status_message, 
                   u.online_status, c.nickname, c.is_favorite, c.is_blocked
            FROM contacts c
            JOIN users u ON c.contact_id = u.user_id
            WHERE c.user_id = ? AND c.is_blocked = FALSE
            ORDER BY c.is_favorite DESC, u.username
        """, (user_id,))
        
        rows = cursor.fetchall()
        conn.close()
        
        contacts = []
        for row in rows:
            contacts.append({
                'user_id': row[0],
                'username': row[1],
                'avatar': row[2],
                'status_message': row[3],
                'online_status': row[4],
                'nickname': row[5],
                'is_favorite': row[6],
                'is_blocked': row[7]
            })
        
        return contacts

# Authentication Manager
class AuthManager:
    def __init__(self):
        self.secret_key = "your-secret-key-here"  # In production, use environment variable
    
    def generate_token(self, user: User) -> str:
        """Generate JWT token for user"""
        payload = {
            'user_id': user.user_id,
            'username': user.username,
            'exp': datetime.utcnow() + timedelta(hours=24)
        }
        return jwt.encode(payload, self.secret_key, algorithm='HS256')
    
    def verify_token(self, token: str) -> Optional[Dict]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=['HS256'])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None

# Media Handler
class MediaHandler:
    @staticmethod
    def process_image(image_data: bytes) -> str:
        """Process and optimize image"""
        # In production, implement image compression and upload to cloud storage
        # Return URL or base64 encoded string
        return base64.b64encode(image_data).decode()
    
    @staticmethod
    def process_video(video_data: bytes) -> str:
        """Process and optimize video"""
        # In production, implement video compression and upload to cloud storage
        return base64.b64encode(video_data).decode()
    
    @staticmethod
    def process_audio(audio_data: bytes) -> str:
        """Process audio/voice messages"""
        return base64.b64encode(audio_data).decode()

# Real-time Communication Handler (WebSocket simulation)
class RealtimeHandler:
    def __init__(self):
        self.connections = {}
        self.typing_users = {}
    
    def connect_user(self, user_id: str):
        """Connect user to real-time system"""
        self.connections[user_id] = {
            'status': UserStatus.ONLINE,
            'last_activity': datetime.now()
        }
    
    def disconnect_user(self, user_id: str):
        """Disconnect user from real-time system"""
        if user_id in self.connections:
            del self.connections[user_id]
    
    def send_typing_indicator(self, user_id: str, recipient_id: str):
        """Send typing indicator"""
        if recipient_id not in self.typing_users:
            self.typing_users[recipient_id] = []
        self.typing_users[recipient_id].append(user_id)
        
        # Auto-remove after 3 seconds (simulated)
        # In production, use actual timer/scheduler
    
    def broadcast_message(self, message: Message):
        """Broadcast message to recipients"""
        # In production, use WebSocket to send real-time updates
        pass

# UI Components
class UIComponents:
    @staticmethod
    def render_message(message: Message, current_user_id: str, users: Dict[str, User]):
        """Render a single message bubble"""
        is_sent = message.sender_id == current_user_id
        sender = users.get(message.sender_id)
        
        # Message alignment
        col1, col2, col3 = st.columns([1, 8, 1] if is_sent else [1, 8, 1])
        
        with col2:
            # Message container
            message_class = "message-sent" if is_sent else "message-received"
            
            # Sender name (for groups)
            if not is_sent and sender:
                st.caption(f"**{sender.username}**")
            
            # Message content based on type
            if message.message_type == MessageType.TEXT:
                st.markdown(
                    f'<div class="message-bubble {message_class}">{message.content}</div>',
                    unsafe_allow_html=True
                )
            elif message.message_type == MessageType.IMAGE:
                st.image(message.content, width=300)
            elif message.message_type == MessageType.FILE:
                st.download_button(
                    label=f"📎 {message.content}",
                    data=b"",  # File data would be here
                    file_name=message.content
                )
            
            # Message metadata
            status_icon = {
                MessageStatus.SENT: "✓",
                MessageStatus.DELIVERED: "✓✓",
                MessageStatus.READ: "✓✓"
            }.get(message.status, "")
            
            time_str = message.timestamp.strftime("%H:%M")
            st.caption(f"{time_str} {status_icon}")
            
            # Reactions
            if message.reactions:
                reaction_html = '<div class="message-reactions">'
                for emoji, users in message.reactions.items():
                    reaction_html += f'<span class="reaction-chip">{emoji} {len(users)}</span>'
                reaction_html += '</div>'
                st.markdown(reaction_html, unsafe_allow_html=True)
    
    @staticmethod
    def render_chat_list(contacts: List[Dict], active_chat: Optional[str]):
        """Render chat list in sidebar"""
        st.sidebar.markdown("### 💬 Chats")
        
        search = st.sidebar.text_input("🔍 Search conversations", key="chat_search")
        
        # Filter contacts based on search
        filtered_contacts = contacts
        if search:
            filtered_contacts = [
                c for c in contacts 
                if search.lower() in c['username'].lower()
            ]
        
        # Render each chat item
        for contact in filtered_contacts:
            status_class = f"status-{contact['online_status']}"
            active_class = "active" if contact['user_id'] == active_chat else ""
            
            # Create chat item HTML
            chat_html = f"""
            <div class="chat-list-item {active_class}">
                <span class="user-status {status_class}"></span>
                <div>
                    <strong>{contact['username']}</strong>
                    <br>
                    <small>{contact['status_message']}</small>
                </div>
            </div>
            """
            
            if st.sidebar.button(
                contact['username'], 
                key=f"chat_{contact['user_id']}",
                use_container_width=True
            ):
                st.session_state.active_chat = contact['user_id']
                st.rerun()
    
    @staticmethod
    def render_story_bar(stories: List[Story], users: Dict[str, User]):
        """Render stories/status bar"""
        st.markdown("### 📸 Stories")
        
        story_html = '<div class="story-container">'
        
        # Add user's own story
        story_html += """
        <div class="story-item">
            <div class="story-avatar">➕</div>
            <small>Your Story</small>
        </div>
        """
        
        # Add other stories
        for story in stories[:10]:  # Limit to 10 stories
            user = users.get(story.user_id)
            if user:
                story_html += f"""
                <div class="story-item">
                    <div class="story-avatar">👤</div>
                    <small>{user.username}</small>
                </div>
                """
        
        story_html += '</div>'
        st.markdown(story_html, unsafe_allow_html=True)
    
    @staticmethod
    def render_typing_indicator(typing_users: List[str]):
        """Render typing indicator"""
        if typing_users:
            st.markdown("""
            <div class="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
            </div>
            """, unsafe_allow_html=True)

# Main Application
class ChatApplication:
    def __init__(self):
        self.db = DatabaseManager()
        self.auth = AuthManager()
        self.media = MediaHandler()
        self.realtime = RealtimeHandler()
        self.ui = UIComponents()
        
        # Initialize session state
        self.init_session_state()
    
    def init_session_state(self):
        """Initialize Streamlit session state"""
        if 'user' not in st.session_state:
            st.session_state.user = None
        if 'token' not in st.session_state:
            st.session_state.token = None
        if 'active_chat' not in st.session_state:
            st.session_state.active_chat = None
        if 'active_page' not in st.session_state:
            st.session_state.active_page = 'chats'
        if 'messages' not in st.session_state:
            st.session_state.messages = {}
        if 'contacts' not in st.session_state:
            st.session_state.contacts = []
        if 'stories' not in st.session_state:
            st.session_state.stories = []
        if 'groups' not in st.session_state:
            st.session_state.groups = []
        if 'typing_users' not in st.session_state:
            st.session_state.typing_users = []
        if 'show_media_viewer' not in st.session_state:
            st.session_state.show_media_viewer = False
        if 'current_media' not in st.session_state:
            st.session_state.current_media = None
    
    def run(self):
        """Main application entry point"""
        if st.session_state.user is None:
            self.render_auth_page()
        else:
            self.render_main_app()
    
    def render_auth_page(self):
        """Render authentication (login/register) page"""
        st.title("🚀 Welcome to ChatFusion Pro")
        st.markdown("---")
        
        tab1, tab2 = st.tabs(["Login", "Register"])
        
        with tab1:
            self.render_login_form()
        
        with tab2:
            self.render_register_form()
    
    def render_login_form(self):
        """Render login form"""
        st.subheader("🔐 Login to your account")
        
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            remember_me = st.checkbox("Remember me")
            submit_button = st.form_submit_button("Login")
            
            if submit_button:
                if username and password:
                    user = self.db.authenticate_user(username, password)
                    if user:
                        st.session_state.user = user
                        st.session_state.token = self.auth.generate_token(user)
                        self.realtime.connect_user(user.user_id)
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
                else:
                    st.error("Please fill in all fields")
    
    def render_register_form(self):
        """Render registration form"""
        st.subheader("📝 Create new account")
        
        with st.form("register_form"):
            username = st.text_input("Choose a username")
            email = st.text_input("Email address")
            password = st.text_input("Password", type="password")
            confirm_password = st.text_input("Confirm password", type="password")
            accept_terms = st.checkbox("I accept the terms and conditions")
            submit_button = st.form_submit_button("Create Account")
            
            if submit_button:
                if not all([username, email, password, confirm_password]):
                    st.error("Please fill in all fields")
                elif password != confirm_password:
                    st.error("Passwords do not match")
                elif len(password) < 6:
                    st.error("Password must be at least 6 characters")
                elif not accept_terms:
                    st.error("Please accept the terms and conditions")
                else:
                    user = self.db.create_user(username, email, password)
                    if user:
                        st.success("Account created successfully! Please login.")
                    else:
                        st.error("Username or email already exists")
    
    def render_main_app(self):
        """Render main chat application"""
        # Load user data
        self.load_user_data()
        
        # Sidebar navigation
        self.render_sidebar()
        
        # Main content area
        if st.session_state.active_page == 'chats':
            self.render_chat_page()
        elif st.session_state.active_page == 'stories':
            self.render_stories_page()
        elif st.session_state.active_page == 'groups':
            self.render_groups_page()
        elif st.session_state.active_page == 'settings':
            self.render_settings_page()
        elif st.session_state.active_page == 'profile':
            self.render_profile_page()
    
    def load_user_data(self):
        """Load user contacts, messages, etc."""
        if st.session_state.user:
            # Load contacts
            st.session_state.contacts = self.db.get_user_contacts(st.session_state.user.user_id)
            
            # Load messages for active chat
            if st.session_state.active_chat:
                messages = self.db.get_messages(
                    st.session_state.user.user_id, 
                    st.session_state.active_chat
                )
                st.session_state.messages[st.session_state.active_chat] = messages
    
    def render_sidebar(self):
        """Render sidebar with navigation and chat list"""
        with st.sidebar:
            # User profile section
            st.markdown("### 👤 Profile")
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown("👤")  # Avatar placeholder
            with col2:
                st.markdown(f"**{st.session_state.user.username}**")
                st.caption(st.session_state.user.status_message)
            
            st.markdown("---")
            
            # Navigation menu
            st.markdown("### 🧭 Navigation")
            pages = {
                'chats': '💬 Chats',
                'stories': '📸 Stories',
                'groups': '👥 Groups',
                'profile': '👤 Profile',
                'settings': '⚙️ Settings'
            }
            
            for page_key, page_name in pages.items():
                if st.button(page_name, key=f"nav_{page_key}", use_container_width=True):
                    st.session_state.active_page = page_key
                    st.rerun()
            
            st.markdown("---")
            
            # Chat list (only show on chats page)
            if st.session_state.active_page == 'chats':
                self.ui.render_chat_list(st.session_state.contacts, st.session_state.active_chat)
            
            st.markdown("---")
            
            # Logout button
            if st.button("🚪 Logout", use_container_width=True):
                self.logout()
    
    def render_chat_page(self):
        """Render main chat interface"""
        if not st.session_state.active_chat:
            # Welcome screen when no chat is selected
            st.markdown("""
            <div class="chat-container" style="text-align: center; padding: 100px 20px;">
                <h1>💬 Welcome to ChatFusion Pro</h1>
                <p>Select a chat from the sidebar to start messaging</p>
                <p>Features include:</p>
                <ul style="text-align: left; max-width: 400px; margin: 0 auto;">
                    <li>Real-time messaging</li>
                    <li>Media sharing (images, videos, files)</li>
                    <li>Voice messages</li>
                    <li>Stories and status updates</li>
                    <li>Group chats</li>
                    <li>Message reactions</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            return
        
        # Get active chat contact info
        active_contact = next(
            (c for c in st.session_state.contacts if c['user_id'] == st.session_state.active_chat), 
            None
        )
        
        if not active_contact:
            st.error("Contact not found")
            return
        
        # Chat header
        col1, col2, col3 = st.columns([1, 8, 1])
        with col2:
            st.markdown(f"""
            <div class="group-header">
                <h3>💬 {active_contact['username']}</h3>
                <p><span class="user-status status-{active_contact['online_status']}"></span>
                   {active_contact['online_status'].title()} • {active_contact['status_message']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Stories bar
        if st.session_state.stories:
            self.ui.render_story_bar(st.session_state.stories, {})
        
        # Messages area
        messages_container = st.container()
        with messages_container:
            messages = st.session_state.messages.get(st.session_state.active_chat, [])
            if messages:
                for message in messages:
                    self.ui.render_message(
                        message, 
                        st.session_state.user.user_id, 
                        {st.session_state.user.user_id: st.session_state.user}
                    )
            else:
                st.info("No messages yet. Start the conversation!")
        
        # Typing indicator
        if st.session_state.typing_users:
            self.ui.render_typing_indicator(st.session_state.typing_users)
        
        # Message input area
        self.render_message_input()
    
    def render_message_input(self):
        """Render message input area with various options"""
        st.markdown("---")
        
        # File upload options
        col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 8])
        
        with col1:
            uploaded_image = st.file_uploader("📷", type=['png', 'jpg', 'jpeg'], key="image_upload")
            if uploaded_image:
                self.handle_media_upload(uploaded_image, MessageType.IMAGE)
        
        with col2:
            uploaded_video = st.file_uploader("🎥", type=['mp4', 'avi', 'mov'], key="video_upload")
            if uploaded_video:
                self.handle_media_upload(uploaded_video, MessageType.VIDEO)
        
        with col3:
            uploaded_file = st.file_uploader("📎", key="file_upload")
            if uploaded_file:
                self.handle_media_upload(uploaded_file, MessageType.FILE)
        
        with col4:
            if st.button("🎤"):
                st.info("Voice recording feature - Would integrate with browser audio API")
        
        # Text message input
        message_input = st.text_input(
            "Type a message...", 
            key="message_input",
            placeholder="Type your message here..."
        )
        
        # Send button
        col1, col2 = st.columns([8, 1])
        with col2:
            if st.button("Send", type="primary") or (message_input and st.session_state.get('send_message', False)):
                if message_input.strip():
                    self.send_message(message_input)
                    st.session_state.message_input = ""
                    st.rerun()
        
        # Quick reactions
        st.markdown("Quick reactions:")
        reaction_cols = st.columns(8)
        reactions = ["👍", "❤️", "😂", "😮", "😢", "😡", "👏", "🔥"]
        for i, emoji in enumerate(reactions):
            with reaction_cols[i]:
                if st.button(emoji, key=f"reaction_{emoji}"):
                    # Add reaction to last message
                    pass
    
    def handle_media_upload(self, uploaded_file, media_type: MessageType):
        """Handle media file uploads"""
        if uploaded_file:
            file_data = uploaded_file.read()
            processed_data = None
            
            if media_type == MessageType.IMAGE:
                processed_data = self.media.process_image(file_data)
            elif media_type == MessageType.VIDEO:
                processed_data = self.media.process_video(file_data)
            else:
                processed_data = self.media.process_audio(file_data)
            
            # Send media message
            message = self.db.send_message(
                st.session_state.user.user_id,
                st.session_state.active_chat,
                uploaded_file.name,  # Store filename
                media_type
            )
            
            # Update local messages
            if st.session_state.active_chat not in st.session_state.messages:
                st.session_state.messages[st.session_state.active_chat] = []
            
            st.session_state.messages[st.session_state.active_chat].append(message)
            st.success(f"{media_type.value.title()} sent!")
    
    def send_message(self, content: str):
        """Send a text message"""
        if content.strip() and st.session_state.active_chat:
            message = self.db.send_message(
                st.session_state.user.user_id,
                st.session_state.active_chat,
                content
            )
            
            # Update local messages
            if st.session_state.active_chat not in st.session_state.messages:
                st.session_state.messages[st.session_state.active_chat] = []
            
            st.session_state.messages[st.session_state.active_chat].append(message)
            
            # Broadcast to real-time handler
            self.realtime.broadcast_message(message)
    
    def render_stories_page(self):
        """Render stories/status page"""
        st.title("📸 Stories")
        st.markdown("---")
        
        # Create story section
        with st.expander("➕ Create New Story", expanded=False):
            story_text = st.text_area("What's on your mind?", placeholder="Share your story...")
            story_image = st.file_uploader("Add image", type=['png', 'jpg', 'jpeg'])
            
            col1, col2 = st.columns(2)
            with col1:
                expires_in = st.selectbox("Expires in", ["24 hours", "1 week", "Never"])
            with col2:
                is_highlight = st.checkbox("Add to highlights")
            
            if st.button("Share Story"):
                if story_text or story_image:
                    st.success("Story shared!")
                else:
                    st.error("Please add some content to your story")
        
        # Display existing stories
        st.markdown("### Your Stories")
        if not st.session_state.stories:
            st.info("No stories yet. Create your first story!")
        else:
            for story in st.session_state.stories:
                with st.container():
                    st.markdown(f"**Story from {story.created_at.strftime('%Y-%m-%d %H:%M')}**")
                    st.write(story.content)
                    if story.media_url:
                        st.image(story.media_url, width=300)
    
    def render_groups_page(self):
        """Render groups page"""
        st.title("👥 Groups")
        st.markdown("---")
        
        # Create group section
        with st.expander("➕ Create New Group", expanded=False):
            group_name = st.text_input("Group Name")
            group_description = st.text_area("Description")
            
            # Member selection (simplified)
            st.markdown("**Add Members:**")
            available_contacts = [c['username'] for c in st.session_state.contacts]
            selected_members = st.multiselect("Select contacts", available_contacts)
            
            if st.button("Create Group"):
                if group_name and selected_members:
                    # Get member IDs
                    member_ids = [
                        c['user_id'] for c in st.session_state.contacts 
                        if c['username'] in selected_members
                    ]
                    
                    group = self.db.create_group(
                        group_name, 
                        group_description, 
                        st.session_state.user.user_id,
                        member_ids
                    )
                    
                    st.success(f"Group '{group_name}' created successfully!")
                else:
                    st.error("Please provide group name and select at least one member")
        
        # Display existing groups
        st.markdown("### Your Groups")
        if not st.session_state.groups:
            st.info("No groups yet. Create or join a group!")
        else:
            for group in st.session_state.groups:
                with st.container():
                    col1, col2, col3 = st.columns([1, 6, 1])
                    with col1:
                        st.markdown("👥")
                    with col2:
                        st.markdown(f"**{group.name}**")
                        st.caption(group.description)
                        st.caption(f"{len(group.member_ids)} members")
                    with col3:
                        if st.button("Open", key=f"group_{group.group_id}"):
                            st.session_state.active_chat = group.group_id
                            st.session_state.active_page = 'chats'
                            st.rerun()
    
    def render_profile_page(self):
        """Render user profile page"""
        st.title("👤 Profile")
        st.markdown("---")
        
        # Profile editing form
        with st.form("profile_form"):
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.markdown("### Avatar")
                st.markdown("👤")  # Placeholder avatar
                avatar_upload = st.file_uploader("Change avatar", type=['png', 'jpg', 'jpeg'])
            
            with col2:
                st.markdown("### Profile Information")
                username = st.text_input("Username", value=st.session_state.user.username)
                email = st.text_input("Email", value=st.session_state.user.email)
                phone = st.text_input("Phone", value=st.session_state.user.phone or "")
                status_message = st.text_input(
                    "Status Message", 
                    value=st.session_state.user.status_message
                )
                
                online_status = st.selectbox(
                    "Status",
                    ["online", "away", "busy", "offline"],
                    index=["online", "away", "busy", "offline"].index(st.session_state.user.online_status.value)
                )
            
            # Security settings
            st.markdown("### Security")
            col1, col2 = st.columns(2)
            with col1:
                change_password = st.checkbox("Change password")
                if change_password:
                    new_password = st.text_input("New password", type="password")
                    confirm_password = st.text_input("Confirm password", type="password")
            
            with col2:
                two_factor = st.checkbox(
                    "Enable two-factor authentication", 
                    value=st.session_state.user.two_factor_enabled
                )
            
            # Submit button
            if st.form_submit_button("Update Profile"):
                st.success("Profile updated successfully!")
    
    def render_settings_page(self):
        """Render settings page"""
        st.title("⚙️ Settings")
        st.markdown("---")
        
        # Notification settings
        with st.expander("🔔 Notifications", expanded=True):
            st.checkbox("Enable push notifications", value=True)
            st.checkbox("Sound notifications", value=True)
            st.checkbox("Vibration", value=True)
            st.selectbox("Notification tone", ["Default", "Bell", "Chime", "Ding"])
        
        # Privacy settings
        with st.expander("🔒 Privacy", expanded=False):
            st.selectbox("Who can see my profile photo", ["Everyone", "Contacts", "Nobody"])
            st.selectbox("Who can see my status", ["Everyone", "Contacts", "Nobody"])
            st.selectbox("Who can see my last seen", ["Everyone", "Contacts", "Nobody"])
            st.checkbox("Read receipts", value=True)
        
        # Chat settings
        with st.expander("💬 Chat Settings", expanded=False):
            st.selectbox("Theme", ["Light", "Dark", "Auto"])
            st.selectbox("Font size", ["Small", "Medium", "Large"])
            st.slider("Chat backup frequency", 1, 30, 7, help="Days")
            st.checkbox("Auto-download media", value=True)
        
        # Advanced settings
        with st.expander("🔧 Advanced", expanded=False):
            st.selectbox("Language", ["English", "Spanish", "French", "German"])
            st.checkbox("Developer mode", value=False)
            st.button("Clear cache")
            st.button("Export data")
        
        # Danger zone
        with st.expander("⚠️ Danger Zone", expanded=False):
            st.warning("These actions are irreversible!")
            if st.button("Delete all messages", type="secondary"):
                if st.checkbox("I understand this action cannot be undone"):
                    st.error("This feature is not implemented in demo mode")
            
            if st.button("Delete account", type="secondary"):
                if st.checkbox("I want to permanently delete my account"):
                    st.error("This feature is not implemented in demo mode")
    
    def logout(self):
        """Logout user and clear session"""
        if st.session_state.user:
            self.realtime.disconnect_user(st.session_state.user.user_id)
        
        # Clear session state
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        
        st.rerun()

# Application entry point
def main():
    """Main application entry point"""
    app = ChatApplication()
    app.run()

# Run the application
if __name__ == "__main__":
    main()

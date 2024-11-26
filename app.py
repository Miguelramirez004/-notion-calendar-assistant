import os
from dotenv import load_dotenv
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import ChatPromptTemplate
from notion_client import Client
from datetime import datetime, timedelta
import pytz
from typing import List, Dict, Any
import logging
import json
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Data classes and enums
class TaskPriority(Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class TaskStatus(Enum):
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"

@dataclass
class TimeBlock:
    start_time: datetime
    end_time: datetime
    category: str
    energy_level: str
    is_focus_time: bool

@dataclass
class ProductivityMetrics:
    completion_rate: float
    focus_time_hours: float
    task_distribution: Dict[str, int]
    energy_patterns: Dict[str, List[str]]

# Custom CSS
def load_custom_css():
    st.markdown("""
        <style>
        .main { padding: 2rem; }
        .stcard {
            background-color: #ffffff;
            border-radius: 10px;
            padding: 1.5rem;
            margin: 1rem 0;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .header-text {
            color: #1E3A8A;
            font-size: 24px;
            font-weight: 600;
            margin-bottom: 1rem;
        }
        .api-input {
            background-color: #F3F4F6;
            border: 1px solid #E5E7EB;
            border-radius: 6px;
            padding: 0.5rem;
        }
        .chat-message {
            padding: 1rem;
            margin: 0.5rem 0;
            border-radius: 8px;
        }
        .user-message {
            background-color: #E3F2FD;
            margin-left: 2rem;
        }
        .assistant-message {
            background-color: #F3F4F6;
            margin-right: 2rem;
        }
        .event-card {
            background-color: #FFFFFF;
            border-left: 4px solid #2196F3;
            padding: 1rem;
            margin: 0.5rem 0;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            color: #1E3A8A;
        }
        .productivity-card {
            background-color: #f8fafc;
            border-left: 4px solid #4f46e5;
            padding: 1rem;
            margin: 0.5rem 0;
            border-radius: 4px;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }
        .pomodoro-timer {
            font-size: 2.5rem;
            font-weight: bold;
            text-align: center;
            color: #1e293b;
            padding: 1rem;
            background-color: #f1f5f9;
            border-radius: 8px;
            margin: 1rem 0;
        }
        .focus-block {
            background-color: #e0f2fe;
            border: 1px solid #bae6fd;
            padding: 1rem;
            margin: 0.5rem 0;
            border-radius: 4px;
        }
        .task-priority-high { border-left: 4px solid #ef4444; }
        .task-priority-medium { border-left: 4px solid #f59e0b; }
        .task-priority-low { border-left: 4px solid #10b981; }
        .metric-card {
            background-color: #ffffff;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
            margin: 0.5rem 0;
        }
        .timer-controls {
            display: flex;
            justify-content: center;
            gap: 1rem;
            margin: 1rem 0;
        }
        </style>
    """, unsafe_allow_html=True)

class NotionCalendarRAG:
    def __init__(self, openai_api_key: str, notion_api_key: str, database_id: str):
        self.openai_api_key = openai_api_key
        self.notion_api_key = notion_api_key
        self.database_id = database_id
        
        # Initialize productivity tracking
        self.time_blocks: List[TimeBlock] = []
        self.task_queue: List[Dict[str, Any]] = []
        self.focus_sessions: List[Dict[str, Any]] = []
        
        try:
            self.notion = Client(auth=notion_api_key)
            self.model = ChatOpenAI(
                openai_api_key=openai_api_key,
                model="gpt-3.5-turbo",
                temperature=0.7,
                max_tokens=500
            )
        except Exception as e:
            logger.error(f"Error initializing clients: {str(e)}")
            raise

        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        self.system_prompt = """You are an intelligent calendar and productivity assistant with access to a user's Notion calendar. 
        Your capabilities include:
        1. Managing calendar events and schedules
        2. Organizing tasks and priorities
        3. Suggesting productivity improvements
        4. Tracking focus time and work patterns
        5. Managing Pomodoro sessions
        Current date and time: {current_time}"""

        self.user_prompt = """
        Calendar Data:
        {calendar_data}

        Chat History:
        {chat_history}

        User Question: {question}

        Please provide a clear and concise answer based on the calendar data and chat history.
        If referring to dates, please specify them clearly.
        If the question cannot be answered with the available data, please say so.
        """

    def get_table_data(self) -> Dict[str, Any]:
        """Fetch and validate data from Notion database."""
        try:
            response = self.notion.databases.query(database_id=self.database_id)
            return response
        except Exception as e:
            logger.error(f"Error fetching Notion data: {str(e)}")
            raise

    def process_calendar_items(self, notion_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process and validate calendar items."""
        events = []
        try:
            for record in notion_data['results']:
                try:
                    properties = record['properties']
                    date_str = self._extract_date(properties.get('Date', {}))
                    parsed_date = None
                    
                    if date_str != "No Date":
                        try:
                            parsed_date = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                            if parsed_date.tzinfo is None:
                                parsed_date = parsed_date.replace(tzinfo=pytz.UTC)
                        except (ValueError, TypeError) as e:
                            logger.warning(f"Date parsing error: {str(e)}")
                            continue

                    event = {
                        "name": self._extract_title(properties.get('Name', {})),
                        "location": self._extract_rich_text(properties.get('Location', {})),
                        "date": parsed_date.isoformat() if parsed_date else "No Date",
                        "metadata": {
                            "last_edited_time": record.get('last_edited_time', ''),
                            "created_time": record.get('created_time', ''),
                            "id": record.get('id', ''),
                            "url": record.get('url', '')
                        }
                    }
                    
                    if self.validate_event_data(event):
                        events.append(event)
                except Exception as e:
                    logger.error(f"Error processing record: {str(e)}")
                    continue
                    
            return events
        except Exception as e:
            logger.error(f"Error processing calendar items: {str(e)}")
            raise

    def create_time_block(self, start_time: datetime, duration: int, category: str, 
                         energy_level: str = "medium", is_focus_time: bool = False) -> TimeBlock:
        """Create a time block for focused work."""
        end_time = start_time + timedelta(minutes=duration)
        
        if self._check_time_block_conflicts(start_time, end_time):
            raise ValueError("Time block conflicts with existing events")
            
        time_block = TimeBlock(
            start_time=start_time,
            end_time=end_time,
            category=category,
            energy_level=energy_level,
            is_focus_time=is_focus_time
        )
        
        self.time_blocks.append(time_block)
        return time_block

    def create_pomodoro_session(self, task_name: str, work_duration: int = 25, 
                              break_duration: int = 5, cycles: int = 4) -> Dict[str, Any]:
        """Create a Pomodoro session."""
        try:
            now = datetime.now(pytz.UTC)
            session = {
                'task_name': task_name,
                'work_duration': work_duration,
                'break_duration': break_duration,
                'cycles': cycles,
                'start_time': now,
                'intervals': []
            }
            
            current_time = now
            for cycle in range(cycles):
                # Add work interval
                session['intervals'].append({
                    'type': 'work',
                    'start': current_time,
                    'end': current_time + timedelta(minutes=work_duration)
                })
                current_time += timedelta(minutes=work_duration)
                
                # Add break interval
                if cycle < cycles - 1:
                    session['intervals'].append({
                        'type': 'break',
                        'start': current_time,
                        'end': current_time + timedelta(minutes=break_duration)
                    })
                    current_time += timedelta(minutes=break_duration)
            
            self.focus_sessions.append(session)
            return session
        except Exception as e:
            logger.error(f"Error creating Pomodoro session: {str(e)}")
            return None

    def generate_calendar_summary(self, events: List[Dict[str, Any]]) -> str:
        """Generate enhanced calendar summary with metadata."""
        try:
            events.sort(key=lambda x: x['date'] if x['date'] != "No Date" else "9999-12-31")
            summary_parts = ["Calendar Events Summary:"]
            
            for event in events:
                if event['date'] != "No Date":
                    try:
                        event_date = datetime.fromisoformat(event['date'])
                        formatted_date = event_date.strftime("%B %d, %Y at %I:%M %p")
                    except (ValueError, TypeError):
                        formatted_date = event['date']
                else:
                    formatted_date = "No Date"

                event_summary = (
                    f"- Event: {event['name']}\n"
                    f"  Location: {event['location']}\n"
                    f"  Date: {formatted_date}\n"
                    f"  Last Edited: {event['metadata']['last_edited_time']}\n"
                )
                summary_parts.append(event_summary)
            
            return "\n".join(summary_parts)
        except Exception as e:
            logger.error(f"Error generating calendar summary: {str(e)}")
            raise

    def analyze_productivity_metrics(self) -> ProductivityMetrics:
        """Analyze productivity metrics."""
        try:
            # Calculate completion rate
            completed_tasks = sum(1 for task in self.task_queue if task.get('status') == 'completed')
            total_tasks = len(self.task_queue) if self.task_queue else 1
            completion_rate = completed_tasks / total_tasks

            # Calculate focus time
            now = datetime.now(pytz.UTC)
            today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
            focus_time = sum(
                (block.end_time - block.start_time).total_seconds() / 3600
                for block in self.time_blocks
                if block.is_focus_time and block.start_time >= today_start
            )

            # Analyze task distribution
            task_distribution = {}
            for task in self.task_queue:
                category = task.get('category', 'Uncategorized')
                task_distribution[category] = task_distribution.get(category, 0) + 1

            return ProductivityMetrics(
                completion_rate=completion_rate,
                focus_time_hours=focus_time,
                task_distribution=task_distribution,
                energy_patterns=self._analyze_energy_patterns()
            )
        except Exception as e:
            logger.error(f"Error analyzing productivity metrics: {str(e)}")
            return ProductivityMetrics(0.0, 0.0, {}, {})

    def suggest_schedule_improvements(self) -> List[str]:
        """Generate schedule improvement suggestions."""
        try:
            metrics = self.analyze_productivity_metrics()
            suggestions = []

            if metrics.completion_rate < 0.7:
                suggestions.append(
                    "Your task completion rate is below target. Consider breaking down tasks "
                    "into smaller, more manageable pieces."
                )

            if metrics.focus_time_hours < 4:
                suggestions.append(
                    "You might benefit from more focused work time. Try scheduling 2-3 "
                    "focused work blocks of 90-120 minutes each day."
                )

            if not suggestions:
                suggestions.append("You're doing well! Keep maintaining your current productivity rhythm.")

            return suggestions
        except Exception as e:
            logger.error(f"Error generating improvements: {str(e)}")
            return ["Unable to generate suggestions at this time."]

    def _check_time_block_conflicts(self, start_time: datetime, end_time: datetime) -> bool:
        """Check for time block conflicts."""
        return any(
            start_time < block.end_time and end_time > block.start_time
            for block in self.time_blocks
        )

    def _analyze_energy_patterns(self) -> Dict[str, List[str]]:
        """Analyze energy patterns."""
        patterns = {
            'high_energy_times': [],
            'low_energy_times': [],
            'most_productive_days': []
        }
        
        for block in self.time_blocks:
            hour = block.start_time.hour
            if block.energy_level == "high":
                patterns['high_energy_times'].append(f"{hour:02d}:00")
            elif block.energy_level == "low":
                patterns['low_energy_times'].append(f"{hour:02d}:00")

        return patterns

    def validate_event_data(self, event: Dict[str, Any]) -> bool:
        """Validate event data."""
        required_fields = ['name', 'location', 'date']
        try:
            if not all(field in event for field in required_fields):
                return False
            
            if event['date'] != "No Date":
                try:
                    datetime.fromisoformat(event['date'])
                except (ValueError, TypeError):
                    return False
            
            return True
        except Exception as e:
            logger.warning(f"Event validation failed: {str(e)}")
            return False

    def _extract_title(self, property_data: Dict[str, Any]) -> str:
        """Extract title from property data."""
        try:
            return property_data.get('title', [{}])[0].get('text', {}).get('content', "No Title")
        except (IndexError, KeyError):
            return "No Title"

    def _extract_rich_text(self, property_data: Dict[str, Any]) -> str:
        """Extract rich text from property data."""
        try:
            return property_data.get('rich_text', [{}])[0].get('text', {}).get('content', "No Location")
        except (IndexError, KeyError):
            return "No Location"

    def _extract_date(self, property_data: Dict[str, Any]) -> str:
        """Extract date from property data."""
        try:
            return property_data.get('date', {}).get('start', "No Date")
        except (IndexError, KeyError):
            return "No Date"

    def ask_question(self, question: str, calendar_summary: str) -> str:
        """Process questions with context and memory."""
        try:
            current_time = datetime.now(pytz.UTC).isoformat()
            prompt = ChatPromptTemplate.from_messages([
                ("system", self.system_prompt),
                ("human", self.user_prompt)
            ])
            
            formatted_prompt = prompt.format(
                current_time=current_time,
                calendar_data=calendar_summary,
                chat_history=str(self.memory.chat_memory.messages),
                question=question
            )
            
            response = self.model.invoke(formatted_prompt)
            self.memory.chat_memory.add_user_message(question)
            self.memory.chat_memory.add_ai_message(response.content)
            
            return response.content
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            return "I apologize, but I encountered an error processing your question. Please try again."

# Callback functions for session state management
def update_pomodoro_settings():
    """Update Pomodoro settings in session state."""
    if 'pomodoro_work_input' in st.session_state:
        st.session_state.pomodoro_work = st.session_state.pomodoro_work_input
    if 'pomodoro_break_input' in st.session_state:
        st.session_state.pomodoro_break = st.session_state.pomodoro_break_input
    if 'pomodoro_cycles_input' in st.session_state:
        st.session_state.pomodoro_cycles = st.session_state.pomodoro_cycles_input

def update_focus_settings():
    """Update focus settings in session state."""
    if 'focus_duration_input' in st.session_state:
        st.session_state.focus_duration = st.session_state.focus_duration_input
    if 'energy_level_input' in st.session_state:
        st.session_state.energy_level = st.session_state.energy_level_input
    if 'focus_categories_input' in st.session_state:
        st.session_state.focus_categories = st.session_state.focus_categories_input

def create_productivity_sidebar():
    """Create a sidebar section for productivity settings."""
    with st.sidebar:
        st.markdown("### Productivity Settings")
        
        # Time Block Settings
        with st.expander("⏰ Time Block Preferences"):
            st.number_input(
                "Default Focus Block Duration (minutes)", 
                min_value=25, 
                max_value=180, 
                value=st.session_state.focus_duration,
                step=5,
                key="focus_duration_input",
                on_change=update_focus_settings
            )
            
            st.selectbox(
                "Preferred Energy Level", 
                ["High", "Medium", "Low"],
                index=["High", "Medium", "Low"].index(st.session_state.energy_level),
                key="energy_level_input",
                on_change=update_focus_settings
            )
            
            st.multiselect(
                "Focus Categories",
                ["Deep Work", "Creative Work", "Administrative", "Learning", "Planning"],
                default=st.session_state.focus_categories,
                key="focus_categories_input",
                on_change=update_focus_settings
            )
        
        # Pomodoro Settings
        with st.expander("🍅 Pomodoro Settings"):
            st.number_input(
                "Work Duration (minutes)", 
                min_value=15, 
                max_value=60, 
                value=st.session_state.pomodoro_work,
                step=5,
                key="pomodoro_work_input",
                on_change=update_pomodoro_settings
            )
            
            st.number_input(
                "Break Duration (minutes)", 
                min_value=5, 
                max_value=30, 
                value=st.session_state.pomodoro_break,
                step=5,
                key="pomodoro_break_input",
                on_change=update_pomodoro_settings
            )
            
            st.number_input(
                "Number of Cycles", 
                min_value=1, 
                max_value=8, 
                value=st.session_state.pomodoro_cycles,
                step=1,
                key="pomodoro_cycles_input",
                on_change=update_pomodoro_settings
            )
def render_calendar_tab():
    """Render the calendar overview tab."""
    calendar_cols = st.columns([2, 1])
    
    with calendar_cols[0]:
        st.markdown('<h3 class="header-text">Calendar Overview</h3>', unsafe_allow_html=True)
        
        try:
            calendar_data = st.session_state.rag_system.get_table_data()
            events = st.session_state.rag_system.process_calendar_items(calendar_data)
            
            # Display upcoming events
            st.markdown("### Upcoming Events")
            current_time = datetime.now(pytz.UTC)
            
            upcoming_events = []
            for event in events:
                if event['date'] != "No Date":
                    try:
                        event_date = datetime.fromisoformat(event['date'].replace('Z', '+00:00'))
                        if event_date.tzinfo is None:
                            event_date = event_date.replace(tzinfo=pytz.UTC)
                        
                        if event_date >= current_time:
                            upcoming_events.append({
                                **event,
                                'parsed_date': event_date
                            })
                    except (ValueError, TypeError) as e:
                        continue

            if upcoming_events:
                upcoming_events.sort(key=lambda x: x['parsed_date'])
                for event in upcoming_events[:5]:
                    formatted_date = event['parsed_date'].strftime("%B %d, %Y at %I:%M %p %Z")
                    st.markdown(f"""
                        <div class="event-card">
                            <strong>{event['name']}</strong><br>
                            <span>📍 {event['location']}</span><br>
                            <span>📅 {formatted_date}</span>
                        </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("No upcoming events found")

        except Exception as e:
            st.error(f"Error displaying calendar overview: {str(e)}")

    with calendar_cols[1]:
        st.markdown("### 📊 Calendar Analytics")
        
        if 'events' in locals():
            # Quick stats
            total_events = len(events)
            upcoming_count = len(upcoming_events) if 'upcoming_events' in locals() else 0
            
            # Display metrics
            stats_cols = st.columns(2)
            stats_cols[0].metric("Total Events", total_events)
            stats_cols[1].metric("Upcoming Events", upcoming_count)
            
            # Calendar analytics
            if events:
                with st.expander("Event Distribution", expanded=True):
                    # Location distribution
                    locations = {}
                    for event in events:
                        location = event['location']
                        if location != "No Location":
                            locations[location] = locations.get(location, 0) + 1
                    
                    if locations:
                        st.markdown("#### Popular Locations")
                        location_data = pd.DataFrame(
                            list(locations.items()),
                            columns=['Location', 'Count']
                        ).sort_values('Count', ascending=False).head(5)
                        st.bar_chart(location_data.set_index('Location'))            

def render_productivity_tab():
    """Render the productivity dashboard tab."""
    st.markdown("### 📊 Productivity Dashboard")
    
    cols = st.columns([2, 1])
    
    with cols[0]:
        # Time Blocking Section
        st.markdown("#### Time Blocking")
        time_block_cols = st.columns([1, 1])
        
        with time_block_cols[0]:
            start_time = st.time_input(
                "Start Time", 
                datetime.now().time()
            )
            category = st.selectbox(
                "Category", 
                st.session_state.focus_categories
            )
        
        with time_block_cols[1]:
            if st.button("Schedule Focus Time"):
                try:
                    now = datetime.now()
                    start_datetime = datetime.combine(now.date(), start_time)
                    
                    if st.session_state.rag_system:
                        block = st.session_state.rag_system.create_time_block(
                            start_time=start_datetime,
                            duration=st.session_state.focus_duration,
                            category=category,
                            energy_level=st.session_state.energy_level.lower(),
                            is_focus_time=True
                        )
                        st.success(f"Focus block scheduled: {block.start_time.strftime('%I:%M %p')} - {block.end_time.strftime('%I:%M %p')}")
                except Exception as e:
                    st.error(str(e))
        
        # Pomodoro Timer
        st.markdown("#### 🍅 Pomodoro Timer")
        if 'pomodoro_active' not in st.session_state:
            st.session_state.pomodoro_active = False
            st.session_state.pomodoro_start = None
        
        if not st.session_state.pomodoro_active:
            task_name = st.text_input("Task Name", key="pomodoro_task")
            if st.button("Start Pomodoro"):
                if st.session_state.rag_system:
                    session = st.session_state.rag_system.create_pomodoro_session(
                        task_name=task_name,
                        work_duration=st.session_state.pomodoro_work,
                        break_duration=st.session_state.pomodoro_break,
                        cycles=st.session_state.pomodoro_cycles
                    )
                    if session:
                        st.session_state.pomodoro_active = True
                        st.session_state.pomodoro_start = datetime.now()
                        st.session_state.current_session = session
                        st.experimental_rerun()
        else:
            if st.button("Stop Pomodoro"):
                st.session_state.pomodoro_active = False
                st.session_state.pomodoro_start = None
                st.session_state.current_session = None
                st.experimental_rerun()
            
            if hasattr(st.session_state, 'current_session'):
                current_time = datetime.now()
                session = st.session_state.current_session
                
                for interval in session['intervals']:
                    if interval['start'] <= current_time <= interval['end']:
                        remaining = (interval['end'] - current_time).total_seconds()
                        mins, secs = divmod(int(remaining), 60)
                        
                        st.markdown(f"""
                            <div class="pomodoro-timer">
                                {mins:02d}:{secs:02d}
                                <div style="font-size: 1rem; color: #64748b;">
                                    {interval['type'].title()} Time
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                        break
    
    with cols[1]:
        st.markdown("### 📈 Metrics & Insights")
        
        if st.session_state.rag_system:
            metrics = st.session_state.rag_system.analyze_productivity_metrics()
            if metrics:
                st.metric("Task Completion Rate", f"{metrics.completion_rate:.1%}")
                st.metric("Focus Time Today", f"{metrics.focus_time_hours:.1f} hours")
                
                if metrics.task_distribution:
                    fig = px.pie(
                        values=list(metrics.task_distribution.values()),
                        names=list(metrics.task_distribution.keys()),
                        title="Task Distribution"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("#### 💡 Suggestions")
                suggestions = st.session_state.rag_system.suggest_schedule_improvements()
                for suggestion in suggestions:
                    st.markdown(f"""
                        <div class="productivity-card">
                            {suggestion}
                        </div>
                    """, unsafe_allow_html=True)

def render_task_management_tab():
    """Render the task management tab."""
    st.markdown("### ✅ Task Management")
    
    # Task Input Form
    with st.form("task_form"):
        task_name = st.text_input("Task Name")
        task_description = st.text_area("Description")
        
        cols = st.columns(3)
        with cols[0]:
            due_date = st.date_input("Due Date")
        with cols[1]:
            priority = st.selectbox("Priority", ["High", "Medium", "Low"])
        with cols[2]:
            category = st.selectbox(
                "Category", 
                ["Work", "Personal", "Learning", "Administrative"]
            )
        
        submit_task = st.form_submit_button("Add Task")
        
        if submit_task and st.session_state.rag_system:
            task = {
                "name": task_name,
                "description": task_description,
                "due_date": due_date.isoformat(),
                "priority": priority.lower(),
                "category": category,
                "status": "not_started"
            }
            
            if not hasattr(st.session_state.rag_system, 'task_queue'):
                st.session_state.rag_system.task_queue = []
            
            st.session_state.rag_system.task_queue.append(task)
            st.success("Task added successfully!")
    
    # Display Tasks
    if st.session_state.rag_system and hasattr(st.session_state.rag_system, 'task_queue'):
        tasks = st.session_state.rag_system.task_queue
        if tasks:
            st.markdown("#### Current Tasks")
            for task in tasks:
                priority_class = f"task-priority-{task['priority']}"
                
                st.markdown(f"""
                    <div class="productivity-card {priority_class}">
                        <strong>{task['name']}</strong><br>
                        <small>Due: {task['due_date']} | Priority: {task['priority'].title()}</small><br>
                        {task['description']}
                    </div>
                """, unsafe_allow_html=True)
                
                cols = st.columns([1, 1, 1])
                with cols[0]:
                    if st.button("✓ Complete", key=f"complete_{task['name']}"):
                        task['status'] = 'completed'
                        st.experimental_rerun()
                with cols[1]:
                    if st.button("▶ Start", key=f"start_{task['name']}"):
                        task['status'] = 'in_progress'
                        st.experimental_rerun()
                with cols[2]:
                    if st.button("🗑 Delete", key=f"delete_{task['name']}"):
                        st.session_state.rag_system.task_queue.remove(task)
                        st.experimental_rerun()
        else:
            st.info("No tasks added yet. Add a task using the form above.")

def main():
    # Initialize Streamlit config
    st.set_page_config(
        page_title="Notion Calendar & Productivity Assistant",
        page_icon="📅",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Load custom CSS
    load_custom_css()

    # Initialize session state variables
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = None
    
    # Initialize productivity-related session state
    if 'focus_duration' not in st.session_state:
        st.session_state.focus_duration = 90
    if 'energy_level' not in st.session_state:
        st.session_state.energy_level = "Medium"
    if 'focus_categories' not in st.session_state:
        st.session_state.focus_categories = ["Deep Work"]
    if 'pomodoro_work' not in st.session_state:
        st.session_state.pomodoro_work = 25
    if 'pomodoro_break' not in st.session_state:
        st.session_state.pomodoro_break = 5
    if 'pomodoro_cycles' not in st.session_state:
        st.session_state.pomodoro_cycles = 4
    if 'pomodoro_active' not in st.session_state:
        st.session_state.pomodoro_active = False
    if 'current_session' not in st.session_state:
        st.session_state.current_session = None
    if 'focus_blocks' not in st.session_state:
        st.session_state.focus_blocks = []

    # Sidebar configuration
    with st.sidebar:
        st.image("https://notion.so/images/logo-ios.png", width=100)
        st.title("Configuration")
        
        # API Configuration Section
        st.markdown("### API Settings")
        with st.expander("Configure API Keys", expanded=not st.session_state.authenticated):
            openai_api_key = st.text_input(
                "OpenAI API Key",
                type="password",
                help="Enter your OpenAI API key to enable the chatbot",
                key="openai_key_input"
            )
            notion_api_key = st.text_input(
                "Notion API Key",
                type="password",
                help="Enter your Notion integration token",
                key="notion_key_input"
            )
            calendar_database_id = st.text_input(
                "Calendar Database ID",
                help="Enter your Notion calendar database ID",
                key="database_id_input"
            )
            
            if st.button("Connect", key="connect_button"):
                with st.spinner("Connecting to services..."):
                    try:
                        st.session_state.rag_system = NotionCalendarRAG(
                            openai_api_key, notion_api_key, calendar_database_id
                        )
                        st.session_state.authenticated = True
                        st.success("Successfully connected!")
                        time.sleep(1)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Connection failed: {str(e)}")

        if st.session_state.authenticated:
            # Add productivity settings to sidebar
            create_productivity_sidebar()
            
            # Chat Settings
            st.markdown("### Chat Settings")
            temperature = st.slider(
                "Response Temperature",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.1,
                key="temperature",
                help="Adjust the creativity of the AI responses"
            )
            if st.session_state.rag_system:
                st.session_state.rag_system.model.temperature = temperature
            
            if st.button("Clear Chat History"):
                st.session_state.chat_history = []
                if st.session_state.rag_system:
                    st.session_state.rag_system.memory.clear()
                st.success("Chat history cleared!")

    # Main content area
    st.title("Find your work-life balance ")
    
    if not st.session_state.authenticated:
        st.markdown("""
            <div class="stcard">
                <h2 class="header-text">Welcome to workwell! 👋</h2>
                <p>To get started, please configure your API keys in the sidebar.</p>
                <ul>
                    <li>Connect your OpenAI account for AI-powered responses</li>
                    <li>Link your Notion calendar for seamless integration</li>
                    <li>Start asking questions about your schedule!</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        return

    # Main interface when authenticated
    try:
        # Create tabs for different sections
        tab1, tab2, tab3, tab4 = st.tabs([
            "💬 Chat", 
            "📅 Calendar", 
            "⚡ Productivity tools",
            "✅ Tasks"
        ])
        
        with tab1:
            # Chat Interface
            chat_cols = st.columns([2, 1])
            
            with chat_cols[0]:
                st.markdown('<h3 class="header-text">Chat Interface</h3>', unsafe_allow_html=True)
                
                # Chat container
                chat_container = st.container()
                with chat_container:
                    for message in st.session_state.chat_history:
                        message_class = "user-message" if message["role"] == "user" else "assistant-message"
                        st.markdown(f"""
                            <div class="chat-message {message_class}">
                                <strong style="color: #1a1a1a;">{'You' if message["role"] == "user" else '🤖 Assistant'}:</strong><br>
                                <span style="color: #1a1a1a;">{message["content"]}</span>
                            </div>
                        """, unsafe_allow_html=True)
                
                # Question input
                st.markdown("### Ask a Question")
                question = st.text_input(
                    "Question",
                    key="question_input",
                    placeholder="Type your question about your calendar here...",
                    label_visibility="collapsed"
                )
                
                # Process new questions
                if question and question not in [msg["content"] for msg in st.session_state.chat_history if msg["role"] == "user"]:
                    with st.spinner("Thinking..."):
                        try:
                            calendar_data = st.session_state.rag_system.get_table_data()
                            events = st.session_state.rag_system.process_calendar_items(calendar_data)
                            calendar_summary = st.session_state.rag_system.generate_calendar_summary(events)
                            
                            answer = st.session_state.rag_system.ask_question(question, calendar_summary)
                            
                            st.session_state.chat_history.extend([
                                {"role": "user", "content": question},
                                {"role": "assistant", "content": answer}
                            ])
                        except Exception as e:
                            st.error(f"Error processing question: {str(e)}")
            
            with chat_cols[1]:
                st.markdown('<h3 class="header-text">Quick Actions</h3>', unsafe_allow_html=True)
                
                if st.button("📊 Show Schedule Overview"):
                    calendar_data = st.session_state.rag_system.get_table_data()
                    events = st.session_state.rag_system.process_calendar_items(calendar_data)
                    calendar_summary = st.session_state.rag_system.generate_calendar_summary(events)
                    st.markdown(f"```\n{calendar_summary}\n```")
                
                if st.button("💡 Get Productivity Tips"):
                    suggestions = st.session_state.rag_system.suggest_schedule_improvements()
                    for suggestion in suggestions:
                        st.markdown(f"""
                            <div class="productivity-card">
                                {suggestion}
                            </div>
                        """, unsafe_allow_html=True)
        
        with tab2:
            # Calendar Overview Tab
            render_calendar_tab() 
        
        with tab3:
            # Productivity Dashboard Tab
            render_productivity_tab()
        
        with tab4:
            # Task Management Tab
            render_task_management_tab()

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        logger.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()            

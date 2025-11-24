from flask import Flask, render_template, redirect, url_for, request, flash
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, SelectMultipleField, FloatField
from wtforms.validators import DataRequired, Email
from datetime import datetime
import threading, webbrowser, time
from jinja2 import DictLoader
from wtforms.widgets import ListWidget, CheckboxInput
from flask import send_from_directory
from wtforms import SelectField
from wtforms import StringField, SubmitField, SelectMultipleField, FloatField, SelectField, BooleanField, FieldList, FormField
from wtforms.validators import DataRequired, Email, Optional
import json
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, login_user, logout_user, login_required, UserMixin, current_user
from flask import jsonify
from wtforms import SelectField, DateTimeLocalField, TextAreaField, SubmitField
import os
from wtforms.validators import DataRequired
from pathlib import Path
from werkzeug.serving import run_simple
from collections import defaultdict
from flask_login import current_user
from werkzeug.utils import secure_filename
from flask_wtf.file import FileField, FileAllowed
import openai
import re 
from flask import jsonify, request
from flask_login import login_required, current_user
from geopy.geocoders import Nominatim
from collections import defaultdict
from flask import send_file
import pandas as pd
from io import BytesIO



print("Current working directory:", os.getcwd())
print("Database path:", os.path.abspath('crm.db'))
app = Flask(__name__, static_folder='static')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///crm.db'

from dotenv import load_dotenv
load_dotenv()
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'fallback-secret')

db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'

class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.Text, nullable=False)
    role = db.Column(db.String(20), default='client')  # default as client

    doctor = db.relationship('Doctor', back_populates='user', uselist=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

class Job(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(100))
    location = db.Column(db.String(100))
    salary = db.Column(db.String(50))
    description = db.Column(db.Text)
    poster_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    latitude = db.Column(db.Float, nullable=True)
    longitude = db.Column(db.Float, nullable=True)

    poster = db.relationship('User', backref='jobs')



class Message(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    sender_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    recipient_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    job_id = db.Column(db.Integer, db.ForeignKey('job.id'), nullable=True)
    doctor_id = db.Column(db.Integer, db.ForeignKey('doctor.id'), nullable=True)
    content = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    read = db.Column(db.Boolean, default=False)
    message_type = db.Column(db.String(50), default='general')  # <-- NEW FIELD

    sender = db.relationship('User', foreign_keys=[sender_id], backref='sent_messages')
    recipient = db.relationship('User', foreign_keys=[recipient_id], backref='received_messages')
    job = db.relationship('Job', backref='messages')
    doctor = db.relationship('Doctor', backref='messages')


# Doctor Registration Form (For Admin)
class DoctorRegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = StringField('Password', validators=[DataRequired()])
    submit = SubmitField('Create Doctor')

# Job Posting Form (For User)
class JobForm(FlaskForm):
    title = StringField('Title', validators=[DataRequired()])
    location = StringField('Location', validators=[DataRequired()])
    salary = StringField('Salary', validators=[DataRequired()])
    description = StringField('Description', validators=[DataRequired()])
    submit = SubmitField('Post Job')

# Database Models Updates

class ClientRegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired(), Email()])
    password = StringField('Password', validators=[DataRequired()])
    submit = SubmitField('Create Client')

def geocode_location(location_str):
    geolocator = Nominatim(user_agent="job-map-geocoder")
    loc = geolocator.geocode(location_str)
    if loc:
        return loc.latitude, loc.longitude
    return None, None


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))
states = ["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]


# Models
@app.context_processor
def inject_user():
    return dict(current_user=current_user)

    
class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = StringField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

class RegisterForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = StringField('Password', validators=[DataRequired()])
    submit = SubmitField('Register')

    
class Doctor(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), unique=True, nullable=False)
    user = db.relationship('User', back_populates='doctor')

    position = db.Column(db.String(10), nullable=False)
    specialty = db.Column(db.String(100))
    subspecialty = db.Column(db.String(100))
    first_name = db.Column(db.String(50))
    last_name = db.Column(db.String(50))
    email = db.Column(db.String(100), unique=True)
    phone = db.Column(db.String(20))
    alt_phone = db.Column(db.String(20))
    city_of_residence = db.Column(db.String(100))
    medical_school = db.Column(db.String(100))
    med_grad_month_year = db.Column(db.String(20))
    residency = db.Column(db.String(100))
    residency_grad_month_year = db.Column(db.String(20))
    fellowship = db.Column(db.Text)
    fellowship_grad_month_year = db.Column(db.Text)
    bachelors = db.Column(db.String(100))
    bachelors_grad_month_year = db.Column(db.String(20))
    msn = db.Column(db.String(100))
    msn_grad_month_year = db.Column(db.String(20))
    dnp = db.Column(db.String(100))
    dnp_grad_month_year = db.Column(db.String(20))
    additional_training = db.Column(db.Text)
    sponsorship_needed = db.Column(db.Boolean)
    malpractice_cases = db.Column(db.Text)
    certification = db.Column(db.String(30))
    certification_specialty_area = db.Column(db.String(100)) 
    clinically_active = db.Column(db.String(30))  # Yes, No, Never clinically active
    last_clinically_active = db.Column(db.String(20), nullable=True)
    emr = db.Column(db.String(100))
    languages = db.Column(db.String(200))
    states_licensed = db.Column(db.Text)
    states_willing_to_work = db.Column(db.Text)
    salary_expectations = db.Column(db.Float)
    joined = db.Column(db.DateTime, default=datetime.utcnow)
    profile_picture = db.Column(db.String(255), nullable=True)








class ScheduledCall(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    doctor_id = db.Column(db.Integer, db.ForeignKey('doctor.id'))
    scheduled_by_id = db.Column(db.Integer, db.ForeignKey('user.id'))
    job_id = db.Column(db.Integer, db.ForeignKey('job.id'), nullable=True)
    datetime = db.Column(db.DateTime, nullable=False)
    reason = db.Column(db.String(255), nullable=True)
    canceled = db.Column(db.Boolean, default=False)
    reschedule_requested = db.Column(db.Boolean, default=False)
    reschedule_note = db.Column(db.String(255), nullable=True)
    reschedule_datetime = db.Column(db.DateTime, nullable=True)
    invite_status = db.Column(db.String(20), default="Pending") # New: Pending/Accepted/Declined

    # CRUCIAL FIX (MUST MATCH EXACTLY)
    doctor = db.relationship('Doctor', backref=db.backref('scheduled_calls', lazy=True))
    
    scheduled_by = db.relationship('User', backref=db.backref('scheduled_calls_scheduled', lazy=True))
    job = db.relationship('Job', backref='scheduled_calls') 

    def __repr__(self):
        return f'<ScheduledCall {self.id} with Doctor {self.doctor_id}>'






class MalpracticeCaseForm(FlaskForm):
    incident_year = StringField('Incident Year')
    outcome = SelectField(
        'Outcome',
        choices=[('Dropped', 'Dropped'), ('Won', 'Won'), ('Settled/Lost', 'Settled/Lost')],
        validators=[Optional()]
    )
    payout_amount = FloatField('Payout Amount', validators=[Optional()])

    class Meta:
        csrf = False

# Forms
class DoctorForm(FlaskForm):
    position = SelectField('Position', choices=[('MD','MD'),('DO','DO'),('NP','NP'),('PA','PA')], validators=[DataRequired()])
    specialty = StringField('Specialty', validators=[DataRequired()])
    subspecialty = StringField('Subspecialty', validators=[Optional()])
    first_name = StringField('First Name', validators=[DataRequired()])
    last_name = StringField('Last Name', validators=[DataRequired()])
    email = StringField('Email', validators=[DataRequired(), Email()])
    phone = StringField('Phone Number', validators=[Optional()])
    alt_phone = StringField('Alternative Phone Number', validators=[Optional()])
    city_of_residence = StringField('City of Residence', validators=[Optional()])

    # MD/DO fields
    medical_school = StringField('Medical School', validators=[Optional()])
    med_grad_month_year = StringField('Medical Graduation (Month/Year)', validators=[Optional()])
    residency = StringField('Residency', validators=[Optional()])
    residency_grad_month_year = StringField('Residency Graduation (Month/Year)', validators=[Optional()])

    num_fellowships = SelectField(
        'Number of Fellowships',
        choices=[(str(i), str(i)) for i in range(0, 11)],
        validators=[DataRequired()]
    )
    fellowship = FieldList(
        StringField('Fellowship'),
        min_entries=1,
        max_entries=10
    )
    fellowship_grad_month_year = FieldList(
        StringField('Fellowship Graduation (Month/Year)'),
        min_entries=1,
        max_entries=10
    )

    # NP/PA fields
    bachelors = StringField('Bachelors Degree', validators=[Optional()])
    bachelors_grad_month_year = StringField('Bachelors Graduation (Month/Year)', validators=[Optional()])
    msn = StringField('Masters of Science in Nursing', validators=[Optional()])
    msn_grad_month_year = StringField('MSN Graduation (Month/Year)', validators=[Optional()])
    dnp = StringField('Doctor of Nursing', validators=[Optional()])
    dnp_grad_month_year = StringField('DNP Graduation (Month/Year)', validators=[Optional()])
    additional_training = StringField('Additional Training', validators=[Optional()])
    sponsorship_needed = BooleanField('Sponsorship Needed?', validators=[Optional()])

    num_malpractice_cases = SelectField(
        'Number of Malpractice Cases',
        choices=[(str(i), str(i)) for i in range(0, 21)],
        validators=[DataRequired()]
    )

    malpractice_cases = FieldList(
        FormField(MalpracticeCaseForm),
        min_entries=1,
        max_entries=15
    )

    certification = SelectField(
        'Certification',
        choices=[
            ('Board Certified', 'Board Certified'),
            ('Board Eligible', 'Board Eligible'),
            ('Not Boarded', 'Not Boarded')
        ],
        validators=[Optional()]
    )
    certification_specialty_area = StringField('Certification Specialty Area', validators=[Optional()])

    clinically_active = SelectField(
        'Clinically Active?',
        choices=[
            ('Yes', 'Yes'),
            ('No', 'No'),
            ('Never clinically active', 'Never clinically active')
        ],
        validators=[DataRequired()]
    )

    last_clinically_active = StringField('Last Clinically Active (Month/Year)', validators=[Optional()])


    emr = StringField('EMR', validators=[Optional()])
    languages = StringField('Languages', validators=[Optional()])

    states_licensed = SelectMultipleField(
        'States Licensed',
        choices=[(state, state) for state in states],
        option_widget=CheckboxInput(),
        widget=ListWidget(prefix_label=False)
    )
    states_willing_to_work = SelectMultipleField(
        'States Willing to Work',
        choices=[(state, state) for state in states],
        option_widget=CheckboxInput(),
        widget=ListWidget(prefix_label=False)
    )

    salary_expectations = FloatField('Salary Expectations', validators=[Optional()])
    profile_picture = FileField('Profile Picture', validators=[
        FileAllowed(['jpg', 'png', 'jpeg'], 'Images only!')
    ])

    submit = SubmitField('Submit')


class ScheduledCallForm(FlaskForm):
    doctor_id = SelectField('Doctor', validators=[DataRequired()])
    datetime = DateTimeLocalField('Call Date & Time', validators=[DataRequired()], format='%Y-%m-%dT%H:%M')
    reason = TextAreaField('Reason for Call', validators=[DataRequired()])
    submit = SubmitField('Schedule Call')
# Add this clearly above your route definitions:

app.jinja_loader = DictLoader({
    'base.html': '''
    <!doctype html>
    <html lang="en">

    <head>
        <title>Healthcare Systems</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/flatpickr/4.6.13/flatpickr.min.css">
        <link href='https://cdn.jsdelivr.net/npm/fullcalendar@6.1.11/index.global.min.css' rel='stylesheet' />
        <script src='https://cdn.jsdelivr.net/npm/fullcalendar@6.1.11/index.global.min.js'></script>
        <link href="https://cdnjs.cloudflare.com/ajax/libs/cropperjs/1.5.13/cropper.min.css" rel="stylesheet"/>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/cropperjs/1.5.13/cropper.min.js"></script>
        <style>
            body {
                background-color: #ffffff;
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            }

            .form-control,
            .form-select {
                border-radius: 10px;
                border: 1px solid #ced4da;
                padding: 12px;
                box-shadow: none;
                transition: border-color 0.3s ease;
            }

            .form-control:focus,
            .form-select:focus {
                border-color: #00a0e9;
                box-shadow: 0 0 0 0.15rem rgba(0, 160, 233, 0.4);
            }

            label {
                font-weight: 600;
                color: #333;
            }

            .btn {
                border-radius: 50px;
                padding: 10px 20px;
            }

            .navbar {
                background-color: #121212 !important;
                box-shadow: none !important;
            }

            .navbar-brand img {
                height: 50px !important;
                width: auto;
                margin-right: 10px;
            }

            .navbar-brand {
                font-size: 1.4rem;
                font-weight: 600;
            }

            .navbar-nav .nav-link {
                font-size: 0.95rem;
                font-weight: 500;
                margin-left: 15px;
            }

            h1, h2, h3, h4 {
                font-weight: 700;
                color: #121212;
            }

            .footer {
                background-color: #f8f9fa;
                padding: 30px 0;
                margin-top: 50px;landing
            }

            .footer a {
                color: #333;
                text-decoration: none;
            }

            .footer a:hover {
                color: #00a0e9;
            }

            .alert {
                border-radius: 10px;
            }
            .leaflet-marker-icon-numbered {
                position: relative;
                width: 38px;
                height: 50px;
                background: transparent;
            }
            .leaflet-marker-icon-numbered .pin-img {
                width: 38px;
                height: 50px;
                display: block;
            }
            .leaflet-marker-icon-numbered .marker-badge {
                position: absolute;
                top: 6px;         /* Adjust down for vertical center of pin head */
                left: 5px;        /* Adjust for horizontal center of pin head */
                background: #fff;
                color: #0066cc;
                border: 2px solid #0066cc;
                border-radius: 50%;
                width: 20px;
                height: 20px;
                text-align: center;
                font-weight: bold;
                font-size: 1em;
                line-height: 16px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.1);
                z-index: 2;
                padding-top: 2px;
            }
            .leaflet-popup-content {
                padding: 0 !important;
                margin: 0 !important;
            }
            .custom-popup {
                max-width: 340px;
                min-width: 220px;
                border-radius: 14px;
                background: #fff;
                box-shadow: 0 8px 36px rgba(0,0,0,0.09), 0 1.5px 6px #0066cc20;
                padding: 0;
                overflow: hidden;
            }
            .custom-popup-header {
                background: #0066cc;
                color: #fff;
                font-weight: bold;
                padding: 14px 18px;
                font-size: 1.08em;
                border-bottom: 1.5px solid #eaf2fb;
            }
            .custom-job-list {
                max-height: 230px;    /* This enables scrolling for long lists */
                overflow-y: auto;
                padding: 10px 0 10px 0;
            }
            .custom-job {
                padding: 9px 18px 9px 18px;
                border-bottom: 1px solid #eaf2fb;
            }
            .custom-job:last-child {
                border-bottom: none;
            }
            .custom-job-title {
                font-weight: 600;
                color: #0066cc;
                font-size: 1em;
                margin-bottom: 0;
            }
            .custom-job-salary {
                color: #333;
                font-size: 0.97em;
            }
            .custom-view-job {
                color: #0066cc;
                font-weight: bold;
                text-decoration: underline;
                font-size: 0.97em;
                margin-top: 4px;
                display: inline-block;
            }
        </style>
    </head>

    <body>
        <nav class="navbar navbar-expand-lg navbar-dark">
            <div class="container">
                <a class="navbar-brand d-flex align-items-center" href="{{ url_for('landing_page') if request.endpoint == 'login' else url_for('dashboard') }}">
                    <img src="{{ url_for('static', filename='jobsdirectmedicalcutright.png') }}" alt="Logo">
                
                </a>
                <div class="navbar-nav ms-auto">
                    {% if current_user.is_authenticated %}

                        {% set unread_count = current_user.received_messages|selectattr('read', 'equalto', False)|list|length %}
                        {% if current_user.role == 'doctor' %}
                            <a class="nav-link text-white" href="{{ url_for('doctor_dashboard') }}">Dashboard</a>
                            <a class="nav-link text-white" href="{{ url_for('doctor_edit_profile') }}">Edit Profile</a>
                            <a class="nav-link text-white" href="{{ url_for('doctor_jobs') }}">Jobs</a>
                            <a class="nav-link text-white position-relative" href="{{ url_for('doctor_inbox') }}">
                                Inbox
                                {% if unread_count > 0 %}
                                <span class="position-absolute top-0 start-100 translate-middle badge rounded-pill bg-danger">
                                    {{ unread_count }}
                                </span>
                                {% endif %}
                            </a>
                            <a class="nav-link text-white" href="{{ url_for('logout') }}">Logout</a>
                            {% if current_user.doctor.profile_picture %}
                                <a href="{{ url_for('doctor_edit_profile') }}">
                                    <img src="{{ url_for('static', filename=current_user.doctor.profile_picture) }}"
                                        alt="Profile"
                                        class="rounded-circle ms-3"
                                        style="height: 40px; width: 40px; object-fit: cover; border: 2px solid #fff;">
                                </a>
                                {% endif %}

                        
    
                            
                        {% elif current_user.role == 'admin' %}
                            <a class="nav-link text-white" href="{{ url_for('register_doctor') }}">Create Doctor Login</a>
                            <a class="nav-link text-white" href="{{ url_for('register_client') }}">Create Client Login</a>
                            <a class="nav-link text-white" href="{{ url_for('add_doctor') }}">Add Doctor</a>
                            <a class="nav-link text-white" href="{{ url_for('doctors') }}">View Doctors</a>
                            <a class="nav-link text-white" href="{{ url_for('post_job') }}">Post Job</a>
                            <a class="nav-link text-white" href="{{ url_for('admin_analytics') }}">Analytics</a>
                            <a class="nav-link text-white position-relative" href="{{ url_for('admin_inbox') }}">
                                Inbox
                                {% if unread_count > 0 %}
                                <span class="position-absolute top-0 start-100 translate-middle badge rounded-pill bg-danger">
                                    {{ unread_count }}
                                </span>
                                {% endif %}
                            </a>
                            <a class="nav-link text-white" href="{{ url_for('logout') }}">Logout</a>
                        {% elif current_user.role == 'client' %}
                            <a class="nav-link text-white" href="{{ url_for('post_job') }}">Post Job</a>
                            <a class="nav-link text-white" href="{{ url_for('client_my_jobs') }}">My Jobs</a>
                            <a class="nav-link text-white" href="{{ url_for('schedule_call') }}">Schedule Call</a>
                            <a class="nav-link text-white" href="{{ url_for('calls') }}">Scheduled Calls</a>
                            <a class="nav-link text-white" href="{{ url_for('doctors') }}">View Doctors</a>
                            <a class="nav-link text-white position-relative" href="{{ url_for('client_inbox') }}">
                                Inbox
                                {% if unread_count > 0 %}
                                <span class="position-absolute top-0 start-100 translate-middle badge rounded-pill bg-danger">
                                    {{ unread_count }}
                                </span>
                                {% endif %}
                            </a>
                            <a class="nav-link text-white" href="{{ url_for('logout') }}">Logout</a>
                        {% else %}
                            <a class="nav-link text-white" href="{{ url_for('logout') }}">Logout</a>
                        {% endif %}
                    {% else %}
                        <a class="nav-link text-white" href="{{ url_for('login') }}">Login</a>
                        <a class="nav-link text-white" href="{{ url_for('create_account') }}">Create Account</a>
                    {% endif %}
                </div>
            </div>
        </nav>

        <main class="container py-5">
            {% with messages = get_flashed_messages(with_categories=true) %}
            {% for category, msg in messages %}
            <div class="alert alert-{{ category }}">{{ msg }}</div>
            {% endfor %}
            {% endwith %}
            {% block content %}{% endblock %}
        </main>

        <footer class="footer text-center">
            <div class="container">
                <span>&copy; {{ current_year }} BECA Staffing Solutions. All rights reserved.</span>
            </div>
        </footer>

        <script src="https://cdn.jsdelivr.net/npm/flatpickr"></script>
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
        <script>
            flatpickr("#datetime", { enableTime: true, dateFormat: "Y-m-d\\TH:i" });
        </script>
    </body>

    </html>
    ''',



    'index.html': '''{% extends "base.html" %}{% block content %}
        <h1>Beca Staffing CRM Dashboard</h1>
        <a class="btn btn-success" href="/add_doctor">Add Doctor</a>
        <a class="btn btn-info" href="/schedule_call">Schedule Call with Doctor</a>
        <a class="btn btn-primary" href="/calls">View Scheduled Calls</a>
        <a class="btn btn-secondary" href="/doctors">View Doctors</a>
    {% endblock %}''',

    'login.html': '''{% extends "base.html" %}
        {% block content %}

        <style>
            body {
                background: linear-gradient(120deg, #eaf2fb 0%, #f8f9fa 100%);
            }
            .login-container {
                max-width: 420px;
                margin: 70px auto 0 auto;
                background: #fff;
                border-radius: 20px;
                box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.18);
                padding: 38px 38px 30px 38px;
                display: flex;
                flex-direction: column;
                align-items: center;
                animation: fadeIn 1s;
            }
            .login-logo {
                width: 90px;
                height: 90px;
                object-fit: contain;
                margin-bottom: 18px;
            }
            .login-title {
                font-weight: 700;
                font-size: 2rem;
                margin-bottom: 12px;
                color: #0066cc;
            }
            .login-subtext {
                color: #555;
                margin-bottom: 24px;
                font-size: 1.08rem;
                text-align: center;
            }
            .form-control {
                border-radius: 12px;
                padding: 12px;
                font-size: 1.08em;
                margin-bottom: 18px;
                background: #f6f8fc;
                border: 1px solid #cfd8dc;
                transition: border-color 0.2s;
            }
            .form-control:focus {
                border-color: #0066cc;
                box-shadow: 0 0 0 0.09rem #7f7dff60;
            }
            .btn-login {
                background: #0066cc;
                color: #fff;
                padding: 13px 0;
                width: 100%;
                border-radius: 50px;
                font-weight: 600;
                font-size: 1.18em;
                box-shadow: 0 2px 8px #0066cc20;
                margin-top: 10px;
                margin-bottom: 6px;
                transition: background 0.15s;
            }
            .btn-login:hover {
                background: #004c99;
            }
            .login-links {
                display: flex;
                justify-content: space-between;
                width: 100%;
                margin-top: 12px;
            }
            .login-links a {
                font-size: 0.97em;
                color: #0066cc;
                text-decoration: none;
                transition: color 0.15s;
            }
            .login-links a:hover {
                color: #004c99;
                text-decoration: underline;
            }
            @media (max-width: 540px) {
                .login-container { padding: 22px 8px 18px 8px; }
            }
            @keyframes fadeIn {
                from { opacity: 0; transform: translateY(30px);}
                to   { opacity: 1; transform: translateY(0);}
            }
        </style>

        <div class="login-container">
            <img src="{{ url_for('static', filename='jobsdirectmedicalcutright.png') }}" alt="Logo" class="login-logo">
            <div class="login-title">Welcome Back</div>
            <div class="login-subtext">Log in to your account to access your dashboard.</div>

            <form method="post">
                {{ form.hidden_tag() }}
                <div class="mb-3">
                    {{ form.username.label(class="form-label") }}
                    {{ form.username(class="form-control", placeholder="Username") }}
                </div>
                <div class="mb-3">
                    {{ form.password.label(class="form-label") }}
                    {{ form.password(class="form-control", placeholder="Password", type="password") }}
                </div>
                {{ form.submit(class="btn btn-login") }}
            </form>

            <div class="login-links mt-2">
                <a href="{{ url_for('create_account') }}">Create Account</a>
                <a href="#">Forgot Password?</a>
            </div>
        </div>

        {% endblock %}''',

    'client_dashboard.html': '''
        {% extends 'base.html' %}

        {% block content %}
        <h2>Client Dashboard</h2>

        <div class="mb-3">
            <a class="btn btn-success" href="{{ url_for('schedule_call') }}">Schedule Call</a>
            <a class="btn btn-primary" href="{{ url_for('calls') }}">Scheduled Calls</a>
            <a class="btn btn-secondary" href="{{ url_for('doctors') }}">View Doctors</a>
            <a class="btn btn-info" href="{{ url_for('client_inbox') }}">Inbox</a>
            <a class="btn btn-warning" href="{{ url_for('post_job') }}">Post Job</a>
            <a class="btn btn-dark" href="{{ url_for('client_my_jobs') }}">My Jobs</a>
            <a class="btn btn-outline-primary" href="{{ url_for('client_analytics') }}">View Analytics</a>
        </div>

        <div id='calendar'></div>

        <div class="mt-4">
            <h3>Reschedule Requests</h3>
            {% if reschedule_requests %}
                <ul class="list-group">
                {% for request in reschedule_requests %}
                    <li class="list-group-item">
                        Doctor: Dr. {{ request.doctor.first_name }} {{ request.doctor.last_name }}<br>
                        Original: {{ request.datetime.strftime('%Y-%m-%d %H:%M') }}<br>
                        Requested: {{ request.reschedule_datetime.strftime('%Y-%m-%d %H:%M') }}<br>
                        Reason: {{ request.reschedule_note }}<br>

                        <form action="{{ url_for('client_handle_reschedule', call_id=request.id) }}" method="post">
                            <textarea name="client_note" class="form-control mt-2" placeholder="Optional note"></textarea>
                            <button type="submit" name="action" value="accept" class="btn btn-success mt-2">Accept</button>
                            <button type="submit" name="action" value="decline" class="btn btn-danger mt-2">Decline</button>
                        </form>
                    </li>
                {% endfor %}
                </ul>
            {% else %}
                <p>No reschedule requests at this time.</p>
            {% endif %}
        </div>

        <script>
        document.addEventListener('DOMContentLoaded', function() {
            var calendarEl = document.getElementById('calendar');
            var calendar = new FullCalendar.Calendar(calendarEl, {
                initialView: 'dayGridMonth',
                headerToolbar: {
                    left: 'prev,next today',
                    center: 'title',
                    right: 'dayGridMonth,timeGridWeek,timeGridDay'
                },
                height: 650,
                events: {{ events | tojson }},
                eventDidMount: function(info) {
                    // Tooltip to show event status on hover
                    let tooltip = new bootstrap.Tooltip(info.el, {
                        title: info.event.extendedProps.status,
                        placement: 'top',
                        trigger: 'hover',
                        container: 'body'
                    });

                    // Strikethrough canceled events
                    if (info.event.extendedProps.status === 'Canceled') {
                        info.el.style.textDecoration = 'line-through';
                    }
                },
                eventClick: function(info) {
                    window.location.href = "/edit_call/" + info.event.id;
                }
            });
            calendar.render();
        });
        </script>

        <!-- Include Bootstrap Tooltip -->
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>

        {% endblock %}''',




    'register_doctor.html' : '''{% extends "base.html" %}
    {% block content %}
        <h2>Create Doctor Account</h2>
        <form method="post">
            {{ form.hidden_tag() }}
            {{ form.username.label }} {{ form.username(class="form-control") }}
            {{ form.email.label }} {{ form.email(class="form-control") }}
            {{ form.password.label }} {{ form.password(class="form-control") }}
            {{ form.submit(class="btn btn-primary mt-3") }}
        </form>
        {% endblock %}''',
    
    'post_job.html': '''{% extends "base.html" %}
    {% block content %}
        <h2>Post a New Job</h2>
        <form method="post">
            {{ form.hidden_tag() }}
            {{ form.title.label }} {{ form.title(class="form-control") }}
            {{ form.location.label }} {{ form.location(class="form-control") }}
            {{ form.salary.label }} {{ form.salary(class="form-control") }}
            {{ form.description.label }} {{ form.description(class="form-control") }}
            {{ form.submit(class="btn btn-success mt-3") }}
            <a href="{{ url_for('scrape_jobs') }}" class="btn btn-warning mt-3">Scrape Jobs from DocCafe</a>
        </form>
        {% endblock %}''',

    'register.html': '''{% extends "base.html" %}
    {% block content %}
        <h2>Register New User</h2>
        <form method="post">
            {{ form.hidden_tag() }}
            <div class="mb-3">
                {{ form.username.label }} {{ form.username(class="form-control") }}
            </div>
            <div class="mb-3">
                {{ form.password.label }} {{ form.password(class="form-control", type="password") }}
            </div>
            {{ form.submit(class="btn btn-success") }}
        </form>
        {% endblock %}''',

    'add_doctor.html': '''{% extends "base.html" %}
        {% block content %}
        <h2>Add New Doctor</h2>

        <form method="post" enctype="multipart/form-data">
            {{ form.hidden_tag() }}

            <div class="mb-3">
                {{ form.profile_picture.label }}
                {{ form.profile_picture(class="form-control", id="profileInput") }}
            </div>

            <!-- Image Preview for Cropping -->
            <div class="text-center mt-3">
                <img id="preview" style="max-width: 300px; display:none; border-radius: 50%;">
            </div>

            <!-- Hidden input to hold cropped base64 image -->
            <input type="hidden" name="cropped_image_data" id="croppedImageData">

            <!-- Button to trigger cropping -->
            <button type="button" class="btn btn-info mt-2" id="cropBtn" style="display:none;">Crop and Save</button>


            <div class="mb-3">{{ form.position.label }} {{ form.position(class="form-select") }}</div>
            <div class="mb-3">{{ form.specialty.label }} {{ form.specialty(class="form-control") }}</div>
            <div class="mb-3">{{ form.subspecialty.label }} {{ form.subspecialty(class="form-control") }}</div>
            <div class="mb-3">{{ form.first_name.label }} {{ form.first_name(class="form-control") }}</div>
            <div class="mb-3">{{ form.last_name.label }} {{ form.last_name(class="form-control") }}</div>
            <div class="mb-3">{{ form.email.label }} {{ form.email(class="form-control") }}</div>
            <div class="mb-3">{{ form.phone.label }} {{ form.phone(class="form-control") }}</div>
            <div class="mb-3">{{ form.alt_phone.label }} {{ form.alt_phone(class="form-control") }}</div>
            <div class="mb-3">{{ form.city_of_residence.label }} {{ form.city_of_residence(class="form-control") }}</div>

            <h4>MD/DO Information</h4>
            <div class="mb-3">{{ form.medical_school.label }} {{ form.medical_school(class="form-control") }}</div>
            <div class="mb-3">{{ form.med_grad_month_year.label }} {{ form.med_grad_month_year(class="form-control") }}</div>
            <div class="mb-3">{{ form.residency.label }} {{ form.residency(class="form-control") }}</div>
            <div class="mb-3">{{ form.residency_grad_month_year.label }} {{ form.residency_grad_month_year(class="form-control") }}</div>

            <h4>Fellowships</h4>
            <div class="mb-3">
                {{ form.num_fellowships.label }} {{ form.num_fellowships(class="form-select", id="num_fellowships") }}
            </div>
            <div id="fellowship_fields">
                {% for fellowship_field, date_field in zip(form.fellowship, form.fellowship_grad_month_year) %}
                <div class="border p-3 mb-3 rounded fellowship-case">
                    <div class="mb-2">{{ fellowship_field.label }} {{ fellowship_field(class="form-control") }}</div>
                    <div class="mb-2">{{ date_field.label }} {{ date_field(class="form-control") }}</div>
                </div>
                {% endfor %}
            </div>

            <h4>NP/PA Information</h4>
            <div class="mb-3">{{ form.bachelors.label }} {{ form.bachelors(class="form-control") }}</div>
            <div class="mb-3">{{ form.bachelors_grad_month_year.label }} {{ form.bachelors_grad_month_year(class="form-control") }}</div>
            <div class="mb-3">{{ form.msn.label }} {{ form.msn(class="form-control") }}</div>
            <div class="mb-3">{{ form.msn_grad_month_year.label }} {{ form.msn_grad_month_year(class="form-control") }}</div>
            <div class="mb-3">{{ form.dnp.label }} {{ form.dnp(class="form-control") }}</div>
            <div class="mb-3">{{ form.dnp_grad_month_year.label }} {{ form.dnp_grad_month_year(class="form-control") }}</div>
            <div class="mb-3">{{ form.additional_training.label }} {{ form.additional_training(class="form-control") }}</div>
            <div class="form-check mb-3">{{ form.sponsorship_needed(class="form-check-input") }} {{ form.sponsorship_needed.label(class="form-check-label") }}</div>

            <h4>Malpractice Cases</h4>
            <div class="mb-3">
                {{ form.num_malpractice_cases.label }} {{ form.num_malpractice_cases(class="form-select", id="num_malpractice_cases") }}
            </div>
            <div id="malpractice_fields">
                {% for case in form.malpractice_cases %}
                <div class="border p-3 mb-3 rounded malpractice-case">
                    <div class="mb-2">{{ case.incident_year.label }} {{ case.incident_year(class="form-control") }}</div>
                    <div class="mb-2">{{ case.outcome.label }} {{ case.outcome(class="form-select") }}</div>
                    <div class="mb-2">{{ case.payout_amount.label }} {{ case.payout_amount(class="form-control") }}</div>
                </div>
                {% endfor %}
            </div>

            <div class="mb-3">{{ form.certification.label }} {{ form.certification(class="form-select") }}</div>
            <div class="mb-3">{{ form.certification_specialty_area.label }} {{ form.certification_specialty_area(class="form-control") }}</div>
            <div class="mb-3">{{ form.clinically_active.label }} {{ form.clinically_active(class="form-select", id="clinically_active") }}</div>
            <div class="mb-3" id="last_active_field" style="display:none;">{{ form.last_clinically_active.label }} {{ form.last_clinically_active(class="form-control") }}</div>
            <div class="mb-3">{{ form.emr.label }} {{ form.emr(class="form-control") }}</div>
            <div class="mb-3">{{ form.languages.label }} {{ form.languages(class="form-control") }}</div>

            <div class="row mb-3">
                <div class="col-md-6">
                    <label class="form-label"><strong>{{ form.states_licensed.label }}</strong></label>
                    <div class="d-flex flex-wrap border rounded p-2" style="max-height:300px; overflow-y:auto;">
                        {% for state in form.states_licensed %}
                        <div class="form-check me-3" style="width:100px;">{{ state(class="form-check-input") }} {{ state.label(class="form-check-label") }}</div>
                        {% endfor %}
                    </div>
                </div>
                <div class="col-md-6">
                    <label class="form-label"><strong>{{ form.states_willing_to_work.label }}</strong></label>
                    <div class="d-flex flex-wrap border rounded p-2" style="max-height:300px; overflow-y:auto;">
                        {% for state in form.states_willing_to_work %}
                        <div class="form-check me-3" style="width:100px;">{{ state(class="form-check-input") }} {{ state.label(class="form-check-label") }}</div>
                        {% endfor %}
                    </div>
                </div>
            </div>

            <div class="mb-3">{{ form.salary_expectations.label }} {{ form.salary_expectations(class="form-control") }}</div>

            {{ form.submit(class="btn btn-success") }}
        </form>

        <script>
            function toggleFields(selectorId, classSelector) {
                const sel = document.getElementById(selectorId),
                    fields = document.querySelectorAll(classSelector);
                sel.addEventListener('change', () => {
                    const n = parseInt(sel.value, 10);
                    fields.forEach((el, i) => el.style.display = i < n ? 'block' : 'none');
                });
                sel.dispatchEvent(new Event('change'));
            }

            toggleFields('num_malpractice_cases', '.malpractice-case');
            toggleFields('num_fellowships', '.fellowship-case');

            document.getElementById('clinically_active').addEventListener('change', function () {
                const selectedOption = this.value;
                const lastActiveDiv = document.getElementById('last_active_field');
                if (selectedOption === 'No') {
                    lastActiveDiv.style.display = 'block';
                } else {
                    lastActiveDiv.style.display = 'none';
                    document.getElementById("last_clinically_active").value = '';
                }
            });
            document.getElementById('clinically_active').dispatchEvent(new Event('change'));
        </script>
        {% endblock %}''',

    'client_my_jobs.html': '''{% extends "base.html" %}
        {% block content %}
        <h2 class="mb-4">My Posted Jobs</h2>

        <form method="get" class="row g-3 mb-4">
            <div class="col-md-4">
                <input type="text" name="keyword" value="{{ keyword }}" class="form-control" placeholder="Enter Job Title / Keyword(s)">
            </div>
            <div class="col-md-4">
                <input type="text" name="location" value="{{ location }}" class="form-control" placeholder="Enter Location">
            </div>
            <div class="col-md-2">
                <button type="submit" class="btn btn-primary w-100">Search</button>
            </div>
            <div class="col-md-2">
                <a href="{{ url_for('client_my_jobs') }}" class="btn btn-secondary w-100">Clear</a>
            </div>
        </form>

        {% if jobs %}
            {% for job in jobs %}
            <div class="card mb-4 shadow-sm">
                <div class="card-body">
                    <h4 class="card-title">{{ job.title }}</h4>
                    <h6 class="card-subtitle text-muted mb-2">{{ job.location }}</h6>
                    <p class="card-text text-truncate" style="max-height: 5.5em; overflow: hidden;">
                        {{ job.description }}
                    </p>
                    <div class="d-flex justify-content-between align-items-center">
                        <small class="text-muted">{{ job.salary }}</small>
                        <a href="{{ url_for('edit_job', job_id=job.id) }}" class="btn btn-sm btn-warning">Edit</a>
                    </div>
                </div>
            </div>
            {% endfor %}
        {% else %}
            <p>No jobs match your criteria.</p>
        {% endif %}

        <div class="d-flex gap-2 mt-4">
            <a href="{{ url_for('post_job') }}" class="btn btn-outline-primary">Post a New Job</a>
            <a href="{{ url_for('client_dashboard') }}" class="btn btn-outline-secondary">Back to Dashboard</a>
        </div>
        {% endblock %}''',

    'schedule_call.html': '''
        {% extends "base.html" %}
        {% block content %}
            <h2>Schedule Call with Doctor</h2>
            <form method="post">
                {{ form.hidden_tag() }}

                <div class="mb-3">
                    {{ form.doctor_id.label }}
                    {{ form.doctor_id(class="form-select") }}
                </div>

                <div class="mb-3">
                    {{ form.datetime.label }}
                    {{ form.datetime(class="form-control", id="datetime") }}
                </div>

                <div class="mb-3">
                    {{ form.reason.label }}
                    {{ form.reason(class="form-control") }}
                </div>

                {{ form.submit(class="btn btn-primary") }}

                <div class="mt-3">
                    <button name="send_invite" value="yes" class="btn btn-info">Send Invite to Doctor</button>
                </div>
            </form>


            <!-- Include Select2 CSS and JS -->
            <link href="https://cdn.jsdelivr.net/npm/select2@4.1.0-rc.0/dist/css/select2.min.css" rel="stylesheet" />
            <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
            <script src="https://cdn.jsdelivr.net/npm/select2@4.1.0-rc.0/dist/js/select2.min.js"></script>

            <script>
                $(document).ready(function() {
                    $('.form-select').select2({
                        placeholder: "Select Doctor (Name | Email | Specialty)",
                        allowClear: true,
                        width: '100%'
                    });
                });
            </script>
        {% endblock %}''',

    'doctor_profile.html': '''{% extends "base.html" %}{% block content %}
        <h2>Doctor Profile: {{ doctor.first_name }} {{ doctor.last_name }}</h2>
        <div class="card shadow p-4 mb-4">
            <h5 class="card-title text-primary">Basic Information</h5>
            <p><strong>Position:</strong> {{ doctor.position }}</p>
            <p><strong>Specialty:</strong> {{ doctor.specialty }}</p>
            <p><strong>Subspecialty:</strong> {{ doctor.subspecialty }}</p>
            <p><strong>Email:</strong> {{ doctor.email }}</p>
            <p><strong>Phone:</strong> {{ doctor.phone }}</p>
            <p><strong>Alternative Phone:</strong> {{ doctor.alt_phone }}</p>
            <p><strong>City of Residence:</strong> {{ doctor.city_of_residence }}</p>
        </div>

        {% if doctor.medical_school %}
        <div class="card shadow p-4 mb-4">
            <h5 class="card-title text-primary">MD/DO Education</h5>
            <p><strong>Medical School:</strong> {{ doctor.medical_school }}</p>
            <p><strong>Graduation:</strong> {{ doctor.med_grad_month_year }}</p>
            <p><strong>Residency:</strong> {{ doctor.residency }}</p>
            <p><strong>Residency Graduation:</strong> {{ doctor.residency_grad_month_year }}</p>
            <p><strong>Fellowships:</strong> {{ doctor.fellowship }}</p>
            <p><strong>Fellowship Graduation:</strong> {{ doctor.fellowship_grad_month_year }}</p>
        </div>
        {% endif %}

        {% if doctor.bachelors %}
        <div class="card shadow p-4 mb-4">
            <h5 class="card-title text-primary">NP/PA Education</h5>
            <p><strong>Bachelors Degree:</strong> {{ doctor.bachelors }}</p>
            <p><strong>Bachelors Graduation:</strong> {{ doctor.bachelors_grad_month_year }}</p>
            <p><strong>MSN:</strong> {{ doctor.msn }}</p>
            <p><strong>MSN Graduation:</strong> {{ doctor.msn_grad_month_year }}</p>
            <p><strong>DNP:</strong> {{ doctor.dnp }}</p>
            <p><strong>DNP Graduation:</strong> {{ doctor.dnp_grad_month_year }}</p>
            <p><strong>Additional Training:</strong> {{ doctor.additional_training }}</p>
            <p><strong>Sponsorship Needed:</strong> {{ 'Yes' if doctor.sponsorship_needed else 'No' }}</p>
        </div>
        {% endif %}

        <div class="card shadow p-4 mb-4">
            <h5 class="card-title text-primary">Licensing & Work Preferences</h5>
            <p><strong>Certification:</strong> {{ doctor.certification }}</p>
            <p><strong>EMR:</strong> {{ doctor.emr }}</p>
            <p><strong>Languages:</strong> {{ doctor.languages }}</p>
            <p><strong>States Licensed:</strong> {{ doctor.states_licensed }}</p>
            <p><strong>States Willing to Work:</strong> {{ doctor.states_willing_to_work }}</p>
            <p><strong>Salary Expectations:</strong> ${{ doctor.salary_expectations }}</p>
        </div>

        <div class="card shadow p-4 mb-4">
            <h5 class="card-title text-primary">Malpractice Cases</h5>
            {% if malpractice_cases %}
                {% for case in malpractice_cases %}
                    <p><strong>Incident Year:</strong> {{ case.incident_year }}</p>
                    <p><strong>Outcome:</strong> {{ case.outcome }}</p>
                    <p><strong>Payout Amount:</strong> ${{ case.payout_amount }}</p>
                    <hr>
                {% endfor %}
            {% else %}
                <p>No malpractice cases reported.</p>
            {% endif %}
        </div>

        <a href="{{ url_for('edit_doctor', doctor_id=doctor.id) }}" class="btn btn-warning">Edit Profile</a>
        <a href="{{ url_for('send_job_to_doctor', doctor_id=doctor.id) }}" class="btn btn-info">Send Job Posting</a>

        {% if current_user.role in ['client', 'admin'] %}
            {% set first_job = current_user.jobs[0] if current_user.jobs|length > 0 else None %}
            {% if first_job %}
                <a href="{{ url_for('send_invite', doctor_id=doctor.id, job_id=first_job.id) }}" class="btn btn-success">Schedule Call</a>
            {% else %}
                <button class="btn btn-secondary" disabled title="Post a job first">Schedule Call</button>
            {% endif %}
        {% endif %}

        <a href="{{ url_for('doctors') }}" class="btn btn-secondary">Back to Doctors List</a>
    {% endblock %}''',

    'register_client.html': '''
        {% extends "base.html" %}
        {% block content %}
            <h2>Create Client Account</h2>
            <form method="post">
                {{ form.hidden_tag() }}
                <div class="mb-3">
                    {{ form.username.label }} {{ form.username(class="form-control") }}
                </div>
                <div class="mb-3">
                    {{ form.email.label }} {{ form.email(class="form-control") }}
                </div>
                <div class="mb-3">
                    {{ form.password.label }} {{ form.password(class="form-control") }}
                </div>
                {{ form.submit(class="btn btn-primary") }}
            </form>
        {% endblock %}''',

    'calls.html': '''{% extends "base.html" %}
        {% block content %}
            <h2>Scheduled Calls with Doctors</h2>

            <div id='calendar'></div>

            <script>
                document.addEventListener('DOMContentLoaded', function() {
                    var calendarEl = document.getElementById('calendar');

                    var calendar = new FullCalendar.Calendar(calendarEl, {
                        initialView: 'dayGridMonth',
                        headerToolbar: {
                            left: 'prev,next today',
                            center: 'title',
                            right: 'dayGridMonth,timeGridWeek,timeGridDay'
                        },
                        height: 650,
                        events: {{ events | tojson }},
                        
                        // Add this block to handle event clicks:
                        eventClick: function(info) {
                            window.location.href = "/edit_call/" + info.event.id;
                        }
                    });

                    calendar.render();
                });
            </script>
        {% endblock %}''',

    'doctor_dashboard.html':'''{% extends "base.html" %}
        {% block content %}
        <h2>Welcome Dr. {{ doctor.first_name }} {{ doctor.last_name }}</h2>

        <div class="mb-3">
            <a class="btn btn-info" href="{{ url_for('doctor_edit_profile') }}">Edit Profile</a>
            <a class="btn btn-secondary" href="{{ url_for('doctor_jobs') }}">Jobs</a>
            <a class="btn btn-success" href="{{ url_for('doctor_inbox') }}">Inbox</a>
            <a class="btn btn-danger" href="{{ url_for('logout') }}">Logout</a>
        </div>

        <h4>Pending Invites</h4>
        {% for call in pending_invites %}
        <div class="alert alert-info">
            Invite from {{ call.scheduled_by.username }} on {{ call.datetime.strftime('%Y-%m-%d %H:%M') }} for "{{ call.reason }}"
            <br>
            <strong>Job:</strong> 
            <a href="{{ url_for('doctor_jobs') }}#job-{{ call.job.id }}">
                {{ call.job.title }}
            </a>
            <form method="post" action="{{ url_for('handle_invite', call_id=call.id) }}">
                <button name="action" value="accept" class="btn btn-success btn-sm">Accept</button>
                <button name="action" value="decline" class="btn btn-danger btn-sm">Decline</button>
            </form>
        </div>
        {% else %}
        <p>No pending invites.</p>
        {% endfor %}

        <div id="calendar"></div>

        <script>
        document.addEventListener('DOMContentLoaded', function() {
            var calendarEl = document.getElementById('calendar');
            var calendar = new FullCalendar.Calendar(calendarEl, {
                initialView: 'dayGridMonth',
                headerToolbar: {
                    left: 'prev,next today',
                    center: 'title',
                    right: 'dayGridMonth,timeGridWeek,timeGridDay'
                },
                height: 650,
                events: {{ events | tojson }},
                eventDidMount: function(info) {
                    // Tooltip showing event status clearly
                    let tooltip = new bootstrap.Tooltip(info.el, {
                        title: info.event.extendedProps.status,
                        placement: 'top',
                        trigger: 'hover',
                        container: 'body'
                    });

                    // Apply strikethrough for canceled meetings
                    if (info.event.extendedProps.status === 'Canceled') {
                        info.el.style.textDecoration = 'line-through';
                    }
                },
                eventClick: function(info) {
                    window.location.href = "/doctor/call/" + info.event.id;
                }
            });
            calendar.render();
        });
        </script>

        <!-- Include Bootstrap JS for Tooltips (ensure not duplicated) -->
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>

        {% endblock %}''',


    'edit_job.html': '''{% extends "base.html" %}
    {% block content %}
        <h2>Edit Job: {{ job.title }}</h2>
        <form method="post">
            {{ form.hidden_tag() }}
            <div class="mb-3">
                {{ form.title.label }} {{ form.title(class="form-control") }}
            </div>
            <div class="mb-3">
                {{ form.location.label }} {{ form.location(class="form-control") }}
            </div>
            <div class="mb-3">
                {{ form.salary.label }} {{ form.salary(class="form-control") }}
            </div>
            <div class="mb-3">
                {{ form.description.label }} {{ form.description(class="form-control") }}
            </div>
            {{ form.submit(class="btn btn-success") }}
        </form>
    {% endblock %}''',

    'doctor_jobs.html': '''{% extends "base.html" %}
        {% block content %}

        <div class="d-flex justify-content-between align-items-center mb-4">
            <h2 class="mb-0">Find Jobs</h2>
            <button class="btn btn-lg btn-info" id="aiSearchBtn" type="button">
                <i class="bi bi-stars"></i> AI Search
            </button>
        </div>

        <!-- Interactive Map -->
        <div id="job-map" style="width:100%;height:420px;border-radius:12px;margin-bottom:36px;"></div>

        <!-- AI Search Modal -->
        <div class="modal fade" id="aiSearchModal" tabindex="-1" aria-labelledby="aiSearchModalLabel" aria-hidden="true">
        <div class="modal-dialog modal-lg modal-dialog-centered">
            <div class="modal-content" style="border-radius:20px;">
            <div class="modal-header">
                <h5 class="modal-title" id="aiSearchModalLabel">Find Your Best Job Match (AI Powered)</h5>
                <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
            </div>
            <form id="aiSearchForm">
                <div class="modal-body">
                <div class="mb-3">
                    <label><b>Lifestyle Description</b></label>
                    <textarea class="form-control" name="lifestyle" rows="2" placeholder="e.g. Quiet suburb, outdoor activities, work/life balance..."></textarea>
                </div>
                <div class="mb-3">
                    <label><b>Job-specific Wants</b></label>
                    <textarea class="form-control" name="wants" rows="2" placeholder="e.g. Research, teaching, high salary, certain procedures..."></textarea>
                </div>
                <div class="mb-3">
                    <label><b>Location Preferences</b></label>
                    <input class="form-control" name="location" placeholder="e.g. Miami, Florida, Northeast, rural, etc.">
                </div>
                </div>
                <div class="modal-footer">
                <button type="submit" class="btn btn-success">Search with AI</button>
                </div>
            </form>
            <div id="aiResultsContainer" class="p-4" style="display:none;"></div>
            </div>
        </div>
        </div>

        <form method="get" class="row g-3 mb-4">
            <div class="col-md-4">
                <input type="text" name="keyword" value="{{ keyword }}" class="form-control" placeholder="Enter Job Title / Keyword(s)">
            </div>
            <div class="col-md-4">
                <input type="text" name="location" value="{{ location }}" class="form-control" placeholder="Enter Location">
            </div>
            <div class="col-md-2">
                <button type="submit" class="btn btn-primary w-100">Search</button>
            </div>
            <div class="col-md-2">
                <a href="{{ url_for('doctor_jobs') }}" class="btn btn-secondary w-100">Clear</a>
            </div>
        </form>

        {% if jobs %}
            {% for job in jobs %}
            <div class="card mb-4 shadow-sm">
                <div class="card-body">
                    <h4 class="card-title">{{ job.title }}</h4>
                    <h6 class="card-subtitle text-muted mb-2">{{ job.location }}</h6>
                    <p class="card-text text-truncate" style="max-height: 5.5em; overflow: hidden;">
                        {{ job.description }}
                    </p>
                    <div class="d-flex justify-content-between align-items-center">
                        <small class="text-muted">{{ job.salary }}</small>
                        <a href="{{ url_for('view_job', job_id=job.id) }}" class="btn btn-sm btn-outline-primary">View Job</a>
                    </div>
                </div>
            </div>
            {% endfor %}
        {% else %}
            <p>No jobs match your criteria.</p>
        {% endif %}

        <div class="d-flex gap-2 mt-4">
            <a href="{{ url_for('doctor_jobs') }}" class="btn btn-outline-primary">← Back to Full Job Board</a>
            <a href="{{ url_for('doctor_dashboard') }}" class="btn btn-outline-secondary">Back to Dashboard</a>
        </div>

        <!-- Leaflet CSS/JS -->
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
        <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
        <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.1/font/bootstrap-icons.css">
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>

        <script>
        const jobMarkers = {{ job_markers|tojson|safe }};

        // Initialize map
        const map = L.map('job-map').setView([37.5, -96], 4);
        L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
            attribution: 'Map &copy; OpenStreetMap contributors'
        }).addTo(map);

        let markerGroup = L.featureGroup();
        jobMarkers.forEach(markerData => {
            if(markerData.lat && markerData.lng) {
                let count = markerData.jobs.length;

                // This uses the default Leaflet blue marker pin SVG (public domain)
                let iconHtml = `
                    <div class="leaflet-marker-icon-numbered">
                        <img class="pin-img" src="https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png">
                        ${count > 1 ? `<div class="marker-badge">${count}</div>` : ""}
                    </div>
                `;

                let icon = L.divIcon({
                    html: iconHtml,
                    className: '', // No extra class, all styles inside
                    iconSize: [38, 50],
                    iconAnchor: [19, 50] // tip of pin is anchor
                });

                let popupHTML = `
                    <div class="custom-popup">
                        <div class="custom-popup-header">
                            ${markerData.jobs.length === 1
                                ? markerData.jobs[0].title
                                : `${markerData.jobs.length} jobs at this location`}
                        </div>
                        <div class="custom-job-list">
                `;

                markerData.jobs.forEach(job => {
                    popupHTML += `
                        <div class="custom-job">
                            <div class="custom-job-title">${job.title}</div>
                            ${job.salary ? `<div class="custom-job-salary">${job.salary}</div>` : ""}
                            <a href="/doctor/job/${job.id}" target="_blank" class="custom-view-job">View Job</a>
                        </div>
                    `;
                });

                popupHTML += `</div></div>`;

                const marker = L.marker([markerData.lat, markerData.lng], {icon: icon}).addTo(markerGroup);
                marker.bindPopup(popupHTML);
            }
        });

        markerGroup.addTo(map);

        // Fit map to bounds if jobs exist
        if (jobMarkers.length > 0) {
            try {
                map.fitBounds(markerGroup.getBounds().pad(0.2));
            } catch (e) {
                // If all jobs have the same lat/lng, fitBounds can fail; ignore
            }
        }

        // AI Modal Script
        document.getElementById('aiSearchBtn').addEventListener('click', function() {
            new bootstrap.Modal(document.getElementById('aiSearchModal')).show();
            document.getElementById('aiResultsContainer').style.display = 'none';
            document.getElementById('aiResultsContainer').innerHTML = '';
        });
        document.getElementById('aiSearchForm').addEventListener('submit', function(e) {
            e.preventDefault();
            const btn = this.querySelector('button[type=submit]');
            btn.disabled = true; btn.innerText = "Searching...";
            const formData = new FormData(this);
            fetch('{{ url_for("doctor_ai_search_jobs") }}', {
                method: 'POST',
                body: formData
            }).then(res => res.json()).then(data => {
                btn.disabled = false; btn.innerText = "Search with AI";
                document.getElementById('aiResultsContainer').style.display = '';
                document.getElementById('aiResultsContainer').innerHTML = data['html'];
            }).catch(() => {
                btn.disabled = false; btn.innerText = "Search with AI";
                document.getElementById('aiResultsContainer').style.display = '';
                document.getElementById('aiResultsContainer').innerHTML = '<div class="alert alert-danger">AI search failed. Try again in a few seconds.</div>';
            });
        });
        </script>
        {% endblock %}''',

    'landing_page.html': '''
        {% extends "base.html" %}
        {% block content %}
        <!-- AOS Animate On Scroll CSS -->
        <link href="https://cdnjs.cloudflare.com/ajax/libs/aos/2.3.4/aos.css" rel="stylesheet" />

        <style>
            .chart-container {
                transition: transform 0.3s ease;
            }
            .chart-container:hover {
                transform: scale(1.05);
            }

            .emoji-grid {
                gap: 10px;
            }

            .emoji-table td {
                font-size: 1.5rem;
                padding: 3px;
                text-align: center;
            }

            :root {
                --brand-teal: #8ecad4;
                --brand-teal-dark: #5aa4b3;
                --brand-teal-soft: #e8f5f7;
                --brand-charcoal: #1f2b2f;
            }

            .hero {
                background: linear-gradient(180deg, #f6fbfc 0%, var(--brand-teal-soft) 100%);
                color: var(--brand-charcoal);
                padding-top: 80px;
                padding-bottom: 40px;
            }

            .highlight {
                color: var(--brand-teal-dark);
            }

            .btn-custom {
                background-color: var(--brand-teal-dark);
                color: #fff;
                border: none;
                padding: 12px 26px;
                border-radius: 40px;
                box-shadow: 0 10px 30px rgba(90, 164, 179, 0.28);
                transition: transform 0.2s ease, box-shadow 0.2s ease, background-color 0.2s ease;
            }

            .btn-custom:hover {
                background-color: var(--brand-teal);
                transform: translateY(-2px);
                box-shadow: 0 14px 36px rgba(90, 164, 179, 0.35);
            }

            .plus-sign {
                font-size: 3rem;
                color: var(--brand-teal-dark);
                margin-left: 15px;
            }

            .section-alt h5,
            .section-alt p,
            .section-alt small {
                margin: 0;
            }

            @media (max-width: 768px) {
                .plus-sign {
                margin-left: 0;
                margin-top: 10px;
                }
            }
            .emoji-table td {
                transition: transform 0.4s cubic-bezier(.3,1.5,.5,1);
                will-change: transform;
            }
            </style>


        <!-- Hero Section -->
        <div class="hero text-center">
            <div class="container" style="padding-bottom: 200px;">  
                <!-- was 60px-->
                 <img id="hero-logo" src="{{ url_for('static', filename='jobsdirectmedicalcutright.png') }}" alt="DocIV Logo" class="img-fluid mb-4" style="max-width: 500px; transition: transform 0.3s ease;">
                <p class="lead">A direct line from the <span class="highlight">Doctor</span> to the <span class="highlight">Hospital</span>, eliminating annoying recruiters forever.</p>
                <div class="mt-4">
                    <a href="{{ url_for('login') }}" class="btn btn-custom">Physician Login</a>
                    <a href="{{ url_for('login') }}" class="btn btn-custom">Hospital Login</a>
                </div>
            </div>
        </div>

        <!-- Problem Statement -->
          <div class="section text-white" style="background-color: var(--brand-teal-dark); text-align: left; padding: 30px 0;">
            <div class="container">
                <h2 class="mb-4" style="margin-left: 20px;">Cut Out the Noise</h2>
                <p style="max-width: 800px; margin-left: 20px;">
                    Did you know that the average physician is contacted by over 
                    <strong>20 recruiters per month</strong>?<br>
                    <span style="display: inline-block; margin-top: 10px;">
                        DocIV puts an end to this disruption by promising you that your precious data will never be given to any recruiters. 
                        Your profile will only be viewed by verified hospital-employed recruiters.
                    </span>
                </p>
            </div>
        </div>

        <div class="section section-alt text-center" style="padding: 50px 0;">
        <!-- was  100px-->
        <div class="container">
            <h2 class="mb-5">Why DocIV Matters</h2>
            <div class="row justify-content-center g-5 align-items-end">

            <!-- Chart 1: Horizontal Bar -->
            <div class="col-md-4 d-flex flex-column align-items-center">
                <div class="chart-container mb-4">
                <canvas id="chart-overwhelm" width="300" height="180"></canvas>
                </div>
                <h5 class="mt-2">Physician Burnout from Recruiters</h5>
                <p class="fw-bold mb-1">73% feel overwhelmed</p>
                <small class="text-muted">Source: Merritt Hawkins Survey, 2021</small>
            </div>

            <!-- Chart 2: Grid + Plus -->
            <div class="col-md-4 d-flex flex-column align-items-center">
                <div class="emoji-grid mb-4 d-flex align-items-center">
                    <table class="emoji-table" id="emojiTable">
                        {% for i in range(5) %}
                        <tr>
                        {% for j in range(10) %}
                        <td>{% if (i + j) % 2 == 0 %}📧{% else %}📱{% endif %}</td>
                        {% endfor %}
                        </tr>
                        {% endfor %}
                    </table>
                    <div class="plus-sign">+</div>
                </div>
                <h5 class="mt-2">Unwanted Contact Monthly</h5>
                <p class="fw-bold mb-1">50+ cold calls/emails</p>
                <small class="text-muted">Source: Doximity Physician Insights, 2022</small>
            </div>

            <!-- Chart 3: Pie -->
            <div class="col-md-4 d-flex flex-column align-items-center">
                <div class="chart-container mb-4" style="max-width: 230px;">
                <canvas id="chart-direct" width="230" height="230"></canvas>
                </div>
                <h5 class="mt-2">Hospitals Want Direct Communication</h5>
                <p class="fw-bold mb-1">90% prefer to skip agencies</p>
                <small class="text-muted">Source: MGMA Stat Poll, 2023</small>
            </div>

            </div>
        </div>
        </div>

        <!-- How It Works Section -->
        <div class="section text-center">
            <div class="container" style="padding-bottom: 200px;">
            <!-- was  60px-->
                <img src="{{ url_for('static', filename='dociv_workflow.png') }}" alt="DocIV Workflow Graphic" class="img-fluid mt-4 shadow rounded" style="max-width: 500px;">
            </div>
        </div>

        <!-- Call to Action -->
        <div class="section section-alt text-center">
            <div class="container">
                <h2>Ready to Experience DocIV?</h2>
                <a href="{{ url_for('login') }}" class="btn btn-custom mt-3">Get Started</a>
            </div>
        </div>

        <!-- Scripts -->
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/aos/2.3.4/aos.js"></script>
        <script>
            AOS.init({ duration: 800, once: true });

            // Mousemove parallax effect
            document.addEventListener("mousemove", function (e) {
                const logo = document.getElementById("hero-logo");
                if (!logo) return;
                const moveX = (e.clientX / window.innerWidth - 0.5) * 10;
                const moveY = (e.clientY / window.innerHeight - 0.5) * 10;
                logo.style.transform = `translate(${moveX}px, ${moveY}px)`;
            });

            // Horizontal bar chart
            new Chart(document.getElementById('chart-overwhelm'), {
                type: 'bar',
                data: {
                    labels: ['Feel overwhelmed by recruiters'],
                    datasets: [{
                        data: [73],
                        backgroundColor:  '#5aa4b3'
                    }]
                },
                options: {
                    }]
                },
                options: {
                    indexAxis: 'y',
                    scales: {
                        x: {
                            min: 0,
                            max: 100,
                            ticks: {
                                stepSize: 25,
                                callback: val => val + '%'
                            }
                        },
                        y: {
                            display: false
                        }
                    },
                    plugins: {
                        legend: { display: false },
                        tooltip: {
                            callbacks: {
                                label: ctx => ctx.raw + '% of physicians'
                            }
                        }
                    }
                }
            });

            // Pie chart
            new Chart(document.getElementById('chart-direct'), {
                type: 'pie',
                data: {
                    labels: ['Prefer Direct', 'Prefer Recruiters'],
                    datasets: [{
                        data: [90, 10],
                        backgroundColor: ['#5aa4b3', '#d4e9ed']
                    }]
                },
                options: {
                    plugins: {
                        legend: { position: 'bottom' },
                        tooltip: {
                            callbacks: {
                                label: ctx => `${ctx.label}: ${ctx.raw}%`
                            }
                        }
                    }
                }
            });
            (function() {
            const table = document.getElementById('emojiTable');
            if (!table) return;

            const rows = Array.from(table.rows);
            const cols = rows[0] ? rows[0].cells.length : 0;

            // Parent for the grid (should be .emoji-grid or similar)
            const gridParent = table.parentElement;

            function handleMove(e) {
                // Get mouse X,Y relative to gridParent
                let rect = gridParent.getBoundingClientRect();
                let mx = (e.clientX - rect.left) / rect.width;
                let my = (e.clientY - rect.top) / rect.height;

                rows.forEach((row, i) => {
                    Array.from(row.cells).forEach((cell, j) => {
                        let dx = (j / (cols-1)) - mx;
                        let dy = (i / (rows.length-1)) - my;
                        let dist = Math.sqrt(dx*dx + dy*dy);

                        let angle = Math.atan2(dy, dx);
                        let mag = Math.max(0, 0.22 - dist); // lower = more local effect

                        let tx = Math.cos(angle) * mag * 150;
                        let ty = Math.sin(angle) * mag * 90;

                        let wave = Math.sin(Date.now()/350 + i*0.7 + j*0.5) * 2;

                        cell.style.transform = `translate(${tx}px, ${ty + wave}px) scale(1.12)`;
                    });
                });
            }

            function resetGrid() {
                rows.forEach(row => {
                    Array.from(row.cells).forEach(cell => {
                        cell.style.transform = '';
                    });
                });
            }

            gridParent.addEventListener('mousemove', handleMove);
            gridParent.addEventListener('mouseleave', resetGrid);
        })();
        </script>
        {% endblock %}
        ''',

    'admin_analytics.html': '''{% extends "base.html" %}
        {% block content %}
        <h2 class="mb-4">Admin Job Post Analytics</h2>

        <input class="form-control mb-3" type="text" id="jobSearch" placeholder="Search by job title, location, or description...">

        <div id="jobList">
            {% for job in job_data %}
            <div class="card mb-4 shadow-sm job-card">
                <div class="card-body">
                    <h5 class="card-title">{{ job.title }}</h5>
                    <p class="mb-0"><strong>Posted By:</strong> {{ job.client_name }} ({{ job.client_email }})</p>
                    <h6 class="card-subtitle mb-2 text-muted">{{ job.location }}</h6>
                    <p class="card-text text-truncate">{{ job.description }}</p>

                    <p><strong>Interest Count:</strong> {{ job.interest_count }}</p>

                    {% if job.interest_count > 0 %}
                        <a href="{{ url_for('download_job_applicants', job_id=job.id) }}" class="btn btn-sm btn-outline-success mb-2">
                            <i class="bi bi-file-earmark-arrow-down"></i> Download Excel
                        </a>
                        <button class="btn btn-sm btn-outline-info mb-2" data-bs-toggle="collapse" data-bs-target="#doctors-{{ job.id }}">
                            View Interested Doctors
                        </button>
                        <div class="collapse" id="doctors-{{ job.id }}">
                            <ul class="list-group">
                                {% for doc in job.interested_doctors %}
                                <li class="list-group-item d-flex justify-content-between align-items-center">
                                    <span>
                                        {{ doc.name }} – {{ doc.email }}
                                    </span>
                                    <span>
                                        {% if doc.id %}
                                            <a href="{{ url_for('doctor_profile', doctor_id=doc.id) }}"
                                            class="btn btn-sm btn-outline-primary me-2">View Profile</a>
                                            <a href="{{ url_for('send_invite', doctor_id=doc.id, job_id=job.id) }}"
                                            class="btn btn-sm btn-success">Schedule Call</a>
                                        {% else %}
                                            <span class="text-muted">No profile available</span>
                                        {% endif %}
                                    </span>
                                </li>
                                {% endfor %}
                            </ul>
                        </div>
                        <canvas id="chart-{{ job.id }}" height="100"></canvas>
                        <script>
                            const ctx{{ job.id }} = document.getElementById('chart-{{ job.id }}').getContext('2d');
                            new Chart(ctx{{ job.id }}, {
                                type: 'bar',
                                data: {
                                    labels: {{ job.interest_by_day.keys() | list | tojson }},
                                    datasets: [{
                                        label: 'Interest Count by Day',
                                        data: {{ job.interest_by_day.values() | list | tojson }},
                                        backgroundColor: 'rgba(0, 123, 255, 0.5)',
                                        borderColor: 'rgba(0, 123, 255, 1)',
                                        borderWidth: 1
                                    }]
                                },
                                options: {
                                    scales: {
                                        x: { title: { display: true, text: 'Date' } },
                                        y: { beginAtZero: true, title: { display: true, text: 'Interests' } }
                                    }
                                }
                            });
                        </script>
                    {% else %}
                        <p class="text-muted">No interest yet.</p>
                    {% endif %}
                </div>
            </div>
            {% endfor %}
        </div>

        <a href="{{ url_for('admin_dashboard') }}" class="btn btn-secondary mt-4">Back to Dashboard</a>

        <!-- Chart.js + live search -->
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script>
        document.getElementById('jobSearch').addEventListener('keyup', function () {
            const query = this.value.toLowerCase();
            document.querySelectorAll('.job-card').forEach(card => {
                const text = card.innerText.toLowerCase();
                card.style.display = text.includes(query) ? '' : 'none';
            });
        });
        
        </script>
        {% endblock %}''',

    'doctor_edit_profile.html': '''{% extends "base.html" %}
        {% block content %}
        <h2>Edit Doctor: {{ doctor.first_name }} {{ doctor.last_name }}</h2>

        <form method="post" enctype="multipart/form-data">
            {{ form.hidden_tag() }}

            {% if doctor.profile_picture %}
            <div class="mb-3">
                <label><strong>Current Profile Picture:</strong></label><br>
                <img src="{{ url_for('static', filename=doctor.profile_picture) }}" class="img-thumbnail" style="max-width: 150px;">
            </div>
            {% endif %}

            <div class="mb-3">
                {{ form.profile_picture.label }}
                {{ form.profile_picture(class="form-control", id="profileInput") }}
            </div>

            <!-- Image Preview for Cropping -->
            <div class="text-center mt-3" id="crop-container" style="display:none;">
                <div style="display:inline-block; width:300px; height:300px; border-radius:50%; overflow:hidden; background:#f0f0f0; position:relative;">
                    <img id="preview" style="position:absolute; top:0; left:0; min-width:100%; min-height:100%;">
                </div>
            </div>

            <!-- Hidden input to hold cropped base64 image -->
            <input type="hidden" name="cropped_image_data" id="croppedImageData">

            <!-- Button to trigger cropping -->
            <button type="button" class="btn btn-info mt-2" id="cropBtn" style="display:none;">Crop and Save</button>

            <script src="https://cdnjs.cloudflare.com/ajax/libs/cropperjs/1.5.13/cropper.min.js"></script>
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/cropperjs/1.5.13/cropper.min.css"/>

            <script>
            const input = document.getElementById('profileInput');
            const preview = document.getElementById('preview');
            const cropBtn = document.getElementById('cropBtn');
            const cropContainer = document.getElementById('crop-container');
            let cropper;

            input.addEventListener('change', function (e) {
                const file = e.target.files[0];
                if (file) {
                    const reader = new FileReader();
                    reader.onload = function (e) {
                        preview.src = e.target.result;
                        cropContainer.style.display = 'block';
                        preview.style.display = 'block';
                        if (cropper) cropper.destroy();
                        cropper = new Cropper(preview, {
                            aspectRatio: 1,
                            viewMode: 1,
                            autoCropArea: 1,
                            background: false,
                            guides: false,
                            highlight: false,
                            dragMode: 'move',
                            cropBoxMovable: false,
                            cropBoxResizable: false,
                            toggleDragModeOnDblclick: false,
                            ready() {
                                const cropBoxData = cropper.getCropBoxData();
                                const containerData = cropper.getContainerData();
                                cropper.setCropBoxData({
                                    left: (containerData.width - cropBoxData.width) / 2,
                                    top: (containerData.height - cropBoxData.height) / 2,
                                    width: 300,
                                    height: 300
                                });
                            }
                        });
                        cropBtn.style.display = 'inline-block';
                    }
                    reader.readAsDataURL(file);
                }
            });

            cropBtn.addEventListener('click', function () {
                if (cropper) {
                    const canvas = cropper.getCroppedCanvas({
                        width: 300,
                        height: 300,
                        imageSmoothingEnabled: true,
                        imageSmoothingQuality: 'high'
                    });
                    document.getElementById('croppedImageData').value = canvas.toDataURL();
                    cropBtn.innerText = "Image Cropped";
                }
            });
            </script>


            <div class="mb-3">{{ form.position.label }} {{ form.position(class="form-select") }}</div>
            <div class="mb-3">{{ form.specialty.label }} {{ form.specialty(class="form-control") }}</div>
            <div class="mb-3">{{ form.subspecialty.label }} {{ form.subspecialty(class="form-control") }}</div>
            <div class="mb-3">{{ form.first_name.label }} {{ form.first_name(class="form-control") }}</div>
            <div class="mb-3">{{ form.last_name.label }} {{ form.last_name(class="form-control") }}</div>
            <div class="mb-3">{{ form.email.label }} {{ form.email(class="form-control") }}</div>
            <div class="mb-3">{{ form.phone.label }} {{ form.phone(class="form-control") }}</div>
            <div class="mb-3">{{ form.alt_phone.label }} {{ form.alt_phone(class="form-control") }}</div>
            <div class="mb-3">{{ form.city_of_residence.label }} {{ form.city_of_residence(class="form-control") }}</div>

            <h4>MD/DO Information</h4>
            <div class="mb-3">{{ form.medical_school.label }} {{ form.medical_school(class="form-control") }}</div>
            <div class="mb-3">{{ form.med_grad_month_year.label }} {{ form.med_grad_month_year(class="form-control") }}</div>
            <div class="mb-3">{{ form.residency.label }} {{ form.residency(class="form-control") }}</div>
            <div class="mb-3">{{ form.residency_grad_month_year.label }} {{ form.residency_grad_month_year(class="form-control") }}</div>

            <h4>Fellowships</h4>
            <div class="mb-3">
                {{ form.num_fellowships.label }} {{ form.num_fellowships(class="form-select", id="num_fellowships") }}
            </div>
            <div id="fellowship_fields">
                {% for fellowship_field, date_field in zip(form.fellowship, form.fellowship_grad_month_year) %}
                <div class="border p-3 mb-3 rounded fellowship-case">
                    <div class="mb-2">{{ fellowship_field.label }} {{ fellowship_field(class="form-control") }}</div>
                    <div class="mb-2">{{ date_field.label }} {{ date_field(class="form-control") }}</div>
                </div>
                {% endfor %}
            </div>

            <h4>NP/PA Information</h4>
            <div class="mb-3">{{ form.bachelors.label }} {{ form.bachelors(class="form-control") }}</div>
            <div class="mb-3">{{ form.bachelors_grad_month_year.label }} {{ form.bachelors_grad_month_year(class="form-control") }}</div>
            <div class="mb-3">{{ form.msn.label }} {{ form.msn(class="form-control") }}</div>
            <div class="mb-3">{{ form.msn_grad_month_year.label }} {{ form.msn_grad_month_year(class="form-control") }}</div>
            <div class="mb-3">{{ form.dnp.label }} {{ form.dnp(class="form-control") }}</div>
            <div class="mb-3">{{ form.dnp_grad_month_year.label }} {{ form.dnp_grad_month_year(class="form-control") }}</div>
            <div class="mb-3">{{ form.additional_training.label }} {{ form.additional_training(class="form-control") }}</div>
            <div class="form-check mb-3">{{ form.sponsorship_needed(class="form-check-input") }} {{ form.sponsorship_needed.label(class="form-check-label") }}</div>

            <h4>Malpractice Cases</h4>
            <div class="mb-3">
                {{ form.num_malpractice_cases.label }} {{ form.num_malpractice_cases(class="form-select", id="num_malpractice_cases") }}
            </div>
            <div id="malpractice_fields">
                {% for case in form.malpractice_cases %}
                <div class="border p-3 mb-3 rounded malpractice-case">
                    <div class="mb-2">{{ case.incident_year.label }} {{ case.incident_year(class="form-control") }}</div>
                    <div class="mb-2">{{ case.outcome.label }} {{ case.outcome(class="form-select") }}</div>
                    <div class="mb-2">{{ case.payout_amount.label }} {{ case.payout_amount(class="form-control") }}</div>
                </div>
                {% endfor %}
            </div>

            <div class="mb-3">{{ form.certification.label }} {{ form.certification(class="form-select") }}</div>
            <div class="mb-3">{{ form.certification_specialty_area.label }} {{ form.certification_specialty_area(class="form-control") }}</div>
            <div class="mb-3">{{ form.clinically_active.label }} {{ form.clinically_active(class="form-select", id="clinically_active") }}</div>
            <div class="mb-3" id="last_active_field" style="display:none;">{{ form.last_clinically_active.label }} {{ form.last_clinically_active(class="form-control") }}</div>
            <div class="mb-3">{{ form.emr.label }} {{ form.emr(class="form-control") }}</div>
            <div class="mb-3">{{ form.languages.label }} {{ form.languages(class="form-control") }}</div>

            <div class="row mb-3">
                <div class="col-md-6">
                    <label class="form-label"><strong>{{ form.states_licensed.label }}</strong></label>
                    <div class="d-flex flex-wrap border rounded p-2" style="max-height:300px; overflow-y:auto;">
                        {% for state in form.states_licensed %}
                        <div class="form-check me-3" style="width:100px;">{{ state(class="form-check-input") }} {{ state.label(class="form-check-label") }}</div>
                        {% endfor %}
                    </div>
                </div>
                <div class="col-md-6">
                    <label class="form-label"><strong>{{ form.states_willing_to_work.label }}</strong></label>
                    <div class="d-flex flex-wrap border rounded p-2" style="max-height:300px; overflow-y:auto;">
                        {% for state in form.states_willing_to_work %}
                        <div class="form-check me-3" style="width:100px;">{{ state(class="form-check-input") }} {{ state.label(class="form-check-label") }}</div>
                        {% endfor %}
                    </div>
                </div>
            </div>

            <div class="mb-3">{{ form.salary_expectations.label }} {{ form.salary_expectations(class="form-control") }}</div>

            {{ form.submit(class="btn btn-success") }}
        </form>

        <script>
        function toggleFields(selectorId, classSelector) {
            const sel = document.getElementById(selectorId),
                fields = document.querySelectorAll(classSelector);
            sel.addEventListener('change', () => {
                const n = parseInt(sel.value, 10);
                fields.forEach((el, i) => el.style.display = i < n ? 'block' : 'none');
            });
            sel.dispatchEvent(new Event('change'));
        }

        // Initialize dynamic field toggles

        toggleFields('num_malpractice_cases', '.malpractice-case');
        toggleFields('num_fellowships', '.fellowship-case');

        // Clinically active logic

        document.getElementById('clinically_active').addEventListener('change', function () {
            const selectedOption = this.value;
            const lastActiveDiv = document.getElementById('last_active_field');
            if (selectedOption === 'No') {
                lastActiveDiv.style.display = 'block';
            } else {
                lastActiveDiv.style.display = 'none';
                document.getElementById("last_clinically_active").value = '';
            }
        });
        document.getElementById('clinically_active').dispatchEvent(new Event('change'));
        </script>
        {% endblock %}
        ''',

    'view_job.html': '''
    {% extends "base.html" %}
    {% block content %}
    <div class="card shadow p-4">
        <h3>{{ job.title }}</h3>
        <p><strong>Location:</strong> {{ job.location }}</p>
        <p><strong>Salary:</strong> {{ job.salary }}</p>
        <p><strong>Description:</strong><br>{{ job.description }}</p>

        {% if already_interested %}
            <div class="alert alert-info mt-3">You've already expressed interest in this job.</div>
            <button class="btn btn-secondary mt-2" disabled>Interest Already Expressed</button>
        {% else %}
            <form method="post">
                <button class="btn btn-success mt-2">Express Interest</button>
            </form>
        {% endif %}

        <a href="{{ url_for('doctor_jobs') }}" class="btn btn-secondary mt-3">Back to Jobs</a>
    </div>
    {% endblock %}''',

    'inbox.html': '''{% extends 'base.html' %}
            {% block content %}
                <h2>{{ title }}</h2>
                <table class="table table-striped">
                    <thead>
                        <tr>
                            <th>From</th>
                            <th>Message</th>
                            <th>Timestamp</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for msg in messages %}
                        <tr>
                            <td>{{ msg.sender.username }}</td>
                            <td>
                                {{ msg.content }}
                                {% if msg.message_type == 'interest' %}
                                    <a href="{{ url_for('send_invite', doctor_id=msg.doctor_id, job_id=msg.job_id) }}" class="btn btn-primary btn-sm">Send Call Invite</a>
                                    <a href="{{ url_for('doctor_profile', doctor_id=msg.doctor_id) }}" class="btn btn-info btn-sm">View Doctor Profile</a>
                                {% endif %}
                            </td>
                            <td>{{ msg.timestamp.strftime('%Y-%m-%d %H:%M') }}</td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            {% endblock %}''',

    'doctor_call_details.html': '''
        {% extends "base.html" %}
        {% block content %}
        <h2>Scheduled Call Details</h2>

        <p><strong>With:</strong> {{ call.scheduled_by.username }} ({{ call.scheduled_by.role }})</p>
        <p><strong>Date & Time:</strong> {{ call.datetime }}</p>
        <p><strong>Reason:</strong> {{ call.reason }}</p>
        <p><strong>Status:</strong> {% if call.canceled %}Canceled{% elif call.reschedule_requested %}Reschedule Requested{% else %}Scheduled{% endif %}</p>

        {% if not call.canceled %}
        <form method="post">
            <button name="action" value="cancel" class="btn btn-danger">Cancel Meeting</button>
        </form>

        <h3 class="mt-4">Request Reschedule</h3>
        <form method="post">
            <div class="mb-3">
                <label>New Date & Time:</label>
                <input type="datetime-local" name="reschedule_datetime" class="form-control" required>
            </div>
            <div class="mb-3">
                <label>Reason for Reschedule:</label>
                <textarea name="reschedule_note" class="form-control"></textarea>
            </div>
            <button name="action" value="reschedule" class="btn btn-warning">Request Reschedule</button>
        </form>
        {% endif %}
        {% endblock %}''',

    'handle_reschedule.html': '''
        {% extends "base.html" %}
        {% block content %}
        <h2>Handle Reschedule Request</h2>

        <p><strong>Doctor:</strong> {{ call.doctor.first_name }} {{ call.doctor.last_name }}</p>
        <p><strong>Current Date & Time:</strong> {{ call.datetime }}</p>
        <p><strong>Requested Date & Time:</strong> {{ call.reschedule_datetime }}</p>
        <p><strong>Reason for Reschedule:</strong> {{ call.reschedule_note }}</p>

        <form method="post">
            <button name="action" value="accept" class="btn btn-success">Accept Request</button>
            <button name="action" value="reject" class="btn btn-danger">Reject Request</button>
        </form>
        {% endblock %}''',

    'create_account.html': '''{% extends "base.html" %}
    {% block content %}
    <h2 class="text-center mb-4">Create Your Account</h2>
    <p class="text-center">Select your role and fill out the form below to join our platform.</p>

    <ul class="nav nav-tabs" id="accountTabs" role="tablist">
        <li class="nav-item" role="presentation">
            <button class="nav-link active" id="doctor-tab" data-bs-toggle="tab" data-bs-target="#doctor" type="button" role="tab">Doctor</button>
        </li>
        <li class="nav-item" role="presentation">
            <button class="nav-link" id="client-tab" data-bs-toggle="tab" data-bs-target="#client" type="button" role="tab">Hospital</button>
        </li>
    </ul>

    <div class="tab-content border p-4" id="accountTabsContent">
        <!-- Doctor Form -->
        <div class="tab-pane fade show active" id="doctor" role="tabpanel">
            <form method="post" action="/public_register_doctor" class="mt-3">
                <div class="mb-3"><input name="first_name" class="form-control" placeholder="First Name"></div>
                <div class="mb-3"><input name="last_name" class="form-control" placeholder="Last Name"></div>
                <div class="mb-3"><input name="specialty" class="form-control" placeholder="Specialty"></div>
                <div class="mb-3"><input name="email" class="form-control" placeholder="Email"></div>
                <div class="mb-3"><input name="username" class="form-control" placeholder="Username"></div>
                <div class="mb-3"><input type="password" name="password" class="form-control" placeholder="Password"></div>
                <div class="mb-3"><input type="password" name="confirm_password" class="form-control" placeholder="Confirm Password"></div>
                <div class="mb-3"><textarea name="reason" class="form-control" placeholder="Reason for joining (optional)"></textarea></div>
                <button type="submit" class="btn btn-success">Create Doctor Account</button>
            </form>
        </div>

        <!-- Client Form -->
        <div class="tab-pane fade" id="client" role="tabpanel">
            <form method="post" action="/public_register_client" class="mt-3">
                <div class="mb-3"><input name="organization_name" class="form-control" placeholder="Hospital or Organization Name"></div>
                <div class="mb-3"><input name="email" class="form-control" placeholder="Email"></div>
                <div class="mb-3"><input name="username" class="form-control" placeholder="Username"></div>
                <div class="mb-3"><input type="password" name="password" class="form-control" placeholder="Password"></div>
                <div class="mb-3"><input type="password" name="confirm_password" class="form-control" placeholder="Confirm Password"></div>
                <div class="mb-3"><textarea name="reason" class="form-control" placeholder="Reason for joining (optional)"></textarea></div>
                <button type="submit" class="btn btn-info">Create Hospital Account</button>
            </form>
        </div>
    </div>

    <div class="text-center mt-4">
        <a href="{{ url_for('login') }}" class="btn btn-secondary">Back to Login</a>
    </div>

    <!-- Bootstrap Tabs -->
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
    {% endblock %}
    ''',

    'doctors.html': '''{% extends "base.html" %}{% block content %}
        <h2>Doctors in System</h2>
        <table class="table table-striped">
            <tr>
                <th>Name</th>
                <th>Email</th>
                <th>Phone</th>
                <th>Specialty</th>
                <th>States Licensed</th>
                <th>States Willing to Work</th>
                <th>Salary Expectations</th>
                <th>Edit</th>
            </tr>
            {% for doctor in doctors %}
            <tr>
                <td><a href="{{ url_for('doctor_profile', doctor_id=doctor.id) }}">{{ doctor.first_name }} {{ doctor.last_name }}</a></td>
                <td>{{ doctor.email }}</td>
                <td>{{ doctor.phone }}</td>
                <td>{{ doctor.specialty }}</td>
                <td>{{ doctor.states_licensed.replace(",", ", ") if doctor.states_licensed else '' }}</td>
                <td>{{ doctor.states_willing_to_work.replace(",", ", ") if doctor.states_willing_to_work else '' }}</td>
                <td>${{ doctor.salary_expectations }}</td>
                <td>
                    <a href="{{ url_for('edit_doctor', doctor_id=doctor.id) }}" class="btn btn-warning btn-sm">Edit</a>
                </td>
            </tr>
            {% endfor %}
        </table>
    {% endblock %}''',

    'edit_call.html': '''{% extends "base.html" %}
    {% block content %}
        <h2>Edit Scheduled Call</h2>
        <form method="post">
            {{ form.hidden_tag() }}

            <div class="mb-3">
                {{ form.doctor_id.label }}
                {{ form.doctor_id(class="form-select") }}
            </div>

            <div class="mb-3">
                {{ form.datetime.label }}
                {{ form.datetime(class="form-control", id="datetime") }}
            </div>

            <div class="mb-3">
                {{ form.reason.label }}
                {{ form.reason(class="form-control") }}
            </div>

            {{ form.submit(class="btn btn-success") }}
        </form>

        <!-- Include Select2 CSS and JS -->
        <link href="https://cdn.jsdelivr.net/npm/select2@4.1.0-rc.0/dist/css/select2.min.css" rel="stylesheet" />
        <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/select2@4.1.0-rc.0/dist/js/select2.min.js"></script>

        <script>
            $(document).ready(function() {
                $('.form-select').select2({
                    placeholder: "Select Doctor (Name | Email | Specialty)",
                    allowClear: true,
                    width: '100%'
                });
            });
            flatpickr("#datetime", {enableTime: true, dateFormat: "Y-m-d\\TH:i"});
        </script>
    {% endblock %}''',

    'send_job.html': '''{% extends 'base.html' %}
    {% block content %}
        <h2>Send Job to {{ doctor.first_name }} {{ doctor.last_name }}</h2>
        <form method="post">
            <div class="mb-3">
                <label>Select Job:</label>
                <select name="job_id" class="form-select">
                    {% for job in jobs %}
                        <option value="{{ job.id }}">{{ job.title }} ({{ job.location }})</option>
                    {% endfor %}
                </select>
            </div>
            <button class="btn btn-success">Send Job</button>
        </form>
        {% endblock %}''',

    'client_analytics.html': '''{% extends "base.html" %}
        {% block content %}
        <h2 class="mb-4">Job Post Analytics</h2>

        <input class="form-control mb-3" type="text" id="jobSearch" placeholder="Search by job title, location, or description...">

        <div id="jobList">
            {% for job in job_data %}
            <div class="card mb-4 shadow-sm job-card">
                <div class="card-body">
                    <h5 class="card-title">{{ job.title }}</h5>
                    <h6 class="card-subtitle mb-2 text-muted">{{ job.location }}</h6>
                    <p class="card-text text-truncate">{{ job.description }}</p>

                    <p><strong>Interest Count:</strong> {{ job.interest_count }}</p>

                    {% if job.interest_count > 0 %}
                    <a href="{{ url_for('download_job_applicants', job_id=job.id) }}" class="btn btn-sm btn-outline-success mb-2">
                        <i class="bi bi-file-earmark-arrow-down"></i> Download Excel
                    </a>
                    <button class="btn btn-sm btn-outline-info mb-2" data-bs-toggle="collapse" data-bs-target="#doctors-{{ job.id }}">View Interested Doctors</button>
                    <div class="collapse" id="doctors-{{ job.id }}">
                        <ul class="list-group">
                            {% for doc in job.interested_doctors %}
                            <li class="list-group-item d-flex justify-content-between align-items-center">
                                <span>
                                    {{ doc.name }} – {{ doc.email }}
                                </span>
                                <span>
                                    {% if doc.id %}
                                        <a href="{{ url_for('doctor_profile', doctor_id=doc.id) }}"
                                        class="btn btn-sm btn-outline-primary me-2">View Profile</a>
                                        <a href="{{ url_for('send_invite', doctor_id=doc.id, job_id=job.id) }}"
                                        class="btn btn-sm btn-success">Schedule Call</a>
                                    {% else %}
                                        <span class="text-muted">No profile available</span>
                                    {% endif %}
                                </span>
                            </li>
                            {% endfor %}
                        </ul>
                    </div>

                    <canvas id="chart-{{ job.id }}" height="100"></canvas>
                    <script>
                        const ctx{{ job.id }} = document.getElementById('chart-{{ job.id }}').getContext('2d');
                        new Chart(ctx{{ job.id }}, {
                            type: 'bar',
                            data: {
                                labels: {{ job.interest_by_day.keys() | list | tojson }},
                                datasets: [{
                                    label: 'Interest Count by Day',
                                    data: {{ job.interest_by_day.values() | list | tojson }},
                                    backgroundColor: 'rgba(0, 123, 255, 0.5)',
                                    borderColor: 'rgba(0, 123, 255, 1)',
                                    borderWidth: 1
                                }]
                            },
                            options: {
                                scales: {
                                    x: { title: { display: true, text: 'Date' } },
                                    y: { beginAtZero: true, title: { display: true, text: 'Interests' } }
                                }
                            }
                        });
                    </script>
                    {% else %}
                    <p class="text-muted">No interest yet.</p>
                    {% endif %}
                </div>
            </div>
            {% endfor %}
        </div>

        <a href="{{ url_for('client_dashboard') }}" class="btn btn-secondary mt-4">Back to Dashboard</a>

        <!-- Chart.js + live search -->
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script>
        document.getElementById('jobSearch').addEventListener('keyup', function () {
            const query = this.value.toLowerCase();
            document.querySelectorAll('.job-card').forEach(card => {
                const text = card.innerText.toLowerCase();
                card.style.display = text.includes(query) ? '' : 'none';
            });
        });
        </script>
        {% endblock %}''',

    'edit_doctor.html': '''{% extends \"base.html\" %}
        {% block content %}
        <h2>Edit Doctor: {{ doctor.first_name }} {{ doctor.last_name }}</h2>
        <form method=\"post\">
            {{ form.hidden_tag() }}

            <div class=\"mb-3\">{{ form.position.label }} {{ form.position(class=\"form-select\") }}</div>
            <div class=\"mb-3\">{{ form.specialty.label }} {{ form.specialty(class=\"form-control\") }}</div>
            <div class=\"mb-3\">{{ form.subspecialty.label }} {{ form.subspecialty(class=\"form-control\") }}</div>
            <div class=\"mb-3\">{{ form.first_name.label }} {{ form.first_name(class=\"form-control\") }}</div>
            <div class=\"mb-3\">{{ form.last_name.label }} {{ form.last_name(class=\"form-control\") }}</div>
            <div class=\"mb-3\">{{ form.email.label }} {{ form.email(class=\"form-control\") }}</div>
            <div class=\"mb-3\">{{ form.phone.label }} {{ form.phone(class=\"form-control\") }}</div>
            <div class=\"mb-3\">{{ form.alt_phone.label }} {{ form.alt_phone(class=\"form-control\") }}</div>
            <div class=\"mb-3\">{{ form.city_of_residence.label }} {{ form.city_of_residence(class=\"form-control\") }}</div>

            <h4>MD/DO Information</h4>
            <div class=\"mb-3\">{{ form.medical_school.label }} {{ form.medical_school(class=\"form-control\") }}</div>
            <div class=\"mb-3\">{{ form.med_grad_month_year.label }} {{ form.med_grad_month_year(class=\"form-control\") }}</div>
            <div class=\"mb-3\">{{ form.residency.label }} {{ form.residency(class=\"form-control\") }}</div>
            <div class=\"mb-3\">{{ form.residency_grad_month_year.label }} {{ form.residency_grad_month_year(class=\"form-control\") }}</div>

            <h4>Fellowships</h4>
            <div class=\"mb-3\">
                {{ form.num_fellowships.label }}
                {{ form.num_fellowships(class=\"form-select\", id=\"num_fellowships\") }}
            </div>

            <div id=\"fellowship_fields\">
                {% for fellowship_field, date_field in zip(form.fellowship, form.fellowship_grad_month_year) %}
                <div class=\"border p-3 mb-3 rounded fellowship-case\">
                    <div class=\"mb-2\">
                        {{ fellowship_field.label }} {{ fellowship_field(class=\"form-control\") }}
                    </div>
                    <div class=\"mb-2\">
                        {{ date_field.label }} {{ date_field(class=\"form-control\") }}
                    </div>
                </div>
                {% endfor %}
            </div>

            <h4>NP/PA Information</h4>
            <div class=\"mb-3\">{{ form.bachelors.label }} {{ form.bachelors(class=\"form-control\") }}</div>
            <div class=\"mb-3\">{{ form.bachelors_grad_month_year.label }} {{ form.bachelors_grad_month_year(class=\"form-control\") }}</div>
            <div class=\"mb-3\">{{ form.msn.label }} {{ form.msn(class=\"form-control\") }}</div>
            <div class=\"mb-3\">{{ form.msn_grad_month_year.label }} {{ form.msn_grad_month_year(class=\"form-control\") }}</div>
            <div class=\"mb-3\">{{ form.dnp.label }} {{ form.dnp(class=\"form-control\") }}</div>
            <div class=\"mb-3\">{{ form.dnp_grad_month_year.label }} {{ form.dnp_grad_month_year(class=\"form-control\") }}</div>
            <div class=\"mb-3\">{{ form.additional_training.label }} {{ form.additional_training(class=\"form-control\") }}</div>
            <div class=\"form-check mb-3\">{{ form.sponsorship_needed(class=\"form-check-input\") }} {{ form.sponsorship_needed.label(class=\"form-check-label\") }}</div>

            <h4>Malpractice Cases</h4>
            <div class=\"mb-3\">
                {{ form.num_malpractice_cases.label }}
                {{ form.num_malpractice_cases(class=\"form-select\", id=\"num_malpractice_cases\") }}
            </div>
            <div id=\"malpractice_fields\">
                {% for case in form.malpractice_cases %}
                <div class=\"border p-3 mb-3 rounded malpractice-case\">
                    <div class=\"mb-2\">{{ case.incident_year.label }} {{ case.incident_year(class=\"form-control\") }}</div>
                    <div class=\"mb-2\">{{ case.outcome.label }} {{ case.outcome(class=\"form-select\") }}</div>
                    <div class=\"mb-2\">{{ case.payout_amount.label }} {{ case.payout_amount(class=\"form-control\") }}</div>
                </div>
                {% endfor %}
            </div>

            <div class=\"mb-3\">{{ form.certification.label }} {{ form.certification(class=\"form-select\") }}</div>
            <div class="mb-3">{{ form.certification_specialty_area.label }} {{ form.certification_specialty_area(class="form-control") }}</div>
            <div class="mb-3">
                {{ form.clinically_active.label }} {{ form.clinically_active(class="form-select", id="clinically_active") }}
            </div>

            <div class="mb-3" id="last_active_field" style="display:none;">
                {{ form.last_clinically_active.label }} {{ form.last_clinically_active(class="form-control") }}
            </div>
            <div class=\"mb-3\">{{ form.emr.label }} {{ form.emr(class=\"form-control\") }}</div>
            <div class=\"mb-3\">{{ form.languages.label }} {{ form.languages(class=\"form-control\") }}</div>

           <div class="row mb-3">
                <div class="col-md-6">
                    <label class="form-label"><strong>{{ form.states_licensed.label }}</strong></label>
                    <div class="d-flex flex-wrap border rounded p-2" style="max-height:300px; overflow-y:auto;">
                        {% for state in form.states_licensed %}
                        <div class="form-check me-3" style="width:100px;">
                            {{ state(class="form-check-input") }} 
                            {{ state.label(class="form-check-label") }}
                        </div>
                        {% endfor %}
                    </div>
                </div>
                <div class="col-md-6">
                    <label class="form-label"><strong>{{ form.states_willing_to_work.label }}</strong></label>
                    <div class="d-flex flex-wrap border rounded p-2" style="max-height:300px; overflow-y:auto;">
                        {% for state in form.states_willing_to_work %}
                        <div class="form-check me-3" style="width:100px;">
                            {{ state(class="form-check-input") }} 
                            {{ state.label(class="form-check-label") }}
                        </div>
                        {% endfor %}
                    </div>
                </div>
            </div>

            <div class=\"mb-3\">{{ form.salary_expectations.label }} {{ form.salary_expectations(class=\"form-control\") }}</div>

            {{ form.submit(class=\"btn btn-success\") }}
        </form>

        
        <script>
            function toggleFields(selectorId, classSelector) {
                const sel = document.getElementById(selectorId),
                    fields = document.querySelectorAll(classSelector);
                sel.addEventListener('change', () => {
                    const n = parseInt(sel.value, 10);
                    fields.forEach((el, i) => el.style.display = i < n ? 'block' : 'none');
                });
                sel.dispatchEvent(new Event('change'));
            }

            toggleFields('num_malpractice_cases', '.malpractice-case');
            toggleFields('num_fellowships', '.fellowship-case');

            // Your new clinically active script below
            document.getElementById('clinically_active').addEventListener('change', function() {
                const selectedOption = this.value;
                const lastActiveDiv = document.getElementById('last_active_field');

                if (selectedOption === 'No') {
                    lastActiveDiv.style.display = 'block';
                } else {
                    lastActiveDiv.style.display = 'none';
                    document.getElementById("last_clinically_active").value = ''; // Clear input if hidden
                }
            });

            // Trigger the event on page load (to maintain state on edit pages)
            document.getElementById('clinically_active').dispatchEvent(new Event('change'));
        </script>
        {% endblock %}'''
         })

# Routes

@app.route('/send_job/<int:doctor_id>', methods=['GET', 'POST'])
@login_required
def send_job_to_doctor(doctor_id):
    if current_user.role not in ['admin', 'client']:
        flash('Unauthorized', 'danger')
        return redirect(url_for('home'))

    doctor = Doctor.query.get_or_404(doctor_id)
    doctor_user = User.query.get(doctor.user_id)

    # Ensure clients can only see their own jobs
    if current_user.role == 'admin':
        jobs = Job.query.all()
    else:  # client
        jobs = Job.query.filter_by(poster_id=current_user.id).all()

    if request.method == 'POST':
        job_id = request.form.get('job_id')
        job = Job.query.get_or_404(job_id)

        # Confirm clients only send their own job posts
        if current_user.role == 'client' and job.poster_id != current_user.id:
            flash('You can only send your own jobs.', 'danger')
            return redirect(url_for('home'))

        message = Message(
            sender_id=current_user.id,
            recipient_id=doctor_user.id,
            job_id=job.id,
            content=f"{current_user.username} has recommended the job: '{job.title}' for you."
        )

        db.session.add(message)
        db.session.commit()
        flash('Job sent to doctor!', 'success')
        return redirect(url_for('doctor_profile', doctor_id=doctor_id))

    return render_template('send_job.html', doctor=doctor, jobs=jobs)



@app.route('/doctor/inbox')
@login_required
def doctor_inbox():
    if current_user.role != 'doctor':
        flash('Unauthorized', 'danger')
        return redirect(url_for('home'))

    messages = Message.query.filter_by(recipient_id=current_user.id).order_by(Message.timestamp.desc()).all()
    
    # Mark unread messages as read
    unread_messages = [msg for msg in messages if not msg.read]
    for msg in unread_messages:
        msg.read = True
    db.session.commit()

    return render_template('inbox.html', messages=messages, title="My Inbox")

@app.route('/admin/inbox')
@login_required
def admin_inbox():
    if current_user.role != 'admin':
        flash('Unauthorized', 'danger')
        return redirect(url_for('home'))

    messages = Message.query.filter_by(recipient_id=current_user.id).order_by(Message.timestamp.desc()).all()

    unread_messages = [msg for msg in messages if not msg.read]
    for msg in unread_messages:
        msg.read = True
    db.session.commit()

    return render_template('inbox.html', messages=messages, title="Admin Inbox")

@app.route('/add_doctor', methods=['GET', 'POST'])
@login_required
def add_doctor():
    if current_user.role != 'admin':
        flash('Unauthorized access!', 'danger')
        return redirect(url_for('home'))

    form = DoctorForm()
    if form.validate_on_submit():
        if User.query.filter_by(email=form.email.data).first() or Doctor.query.filter_by(email=form.email.data).first():
            flash('A user or doctor with this email already exists.', 'danger')
            return redirect(url_for('add_doctor'))

        username_base = form.first_name.data.lower() + form.last_name.data.lower()
        username = username_base
        suffix = 1
        while User.query.filter_by(username=username).first():
            username = f"{username_base}{suffix}"
            suffix += 1

        new_user = User(username=username, email=form.email.data, role='doctor')
        new_user.set_password('TempPass123!')
        db.session.add(new_user)
        db.session.flush()

        profile_path = None
        if form.profile_picture.data:
            picture = form.profile_picture.data
            filename = secure_filename(picture.filename)
            profile_path = f"upload/{filename}"
            picture.save(os.path.join(app.static_folder, profile_path))

        cropped_data = request.form.get('cropped_image_data')
        if cropped_data:
            import base64
            from PIL import Image
            from io import BytesIO

            header, encoded = cropped_data.split(",", 1)
            binary_data = base64.b64decode(encoded)
            img = Image.open(BytesIO(binary_data))

            cropped_filename = f"cropped_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
            cropped_path = os.path.join(app.static_folder, 'upload', cropped_filename)
            img.save(cropped_path)

            profile_path = f"upload/{cropped_filename}"

        new_doctor = Doctor(
            user_id=new_user.id,
            profile_picture=profile_path,
            position=form.position.data,
            specialty=form.specialty.data,
            subspecialty=form.subspecialty.data,
            first_name=form.first_name.data,
            last_name=form.last_name.data,
            email=form.email.data,
            phone=form.phone.data,
            alt_phone=form.alt_phone.data,
            city_of_residence=form.city_of_residence.data,
            medical_school=form.medical_school.data,
            med_grad_month_year=form.med_grad_month_year.data,
            residency=form.residency.data,
            residency_grad_month_year=form.residency_grad_month_year.data,
            fellowship=",".join(form.fellowship.data[:int(form.num_fellowships.data)]),
            fellowship_grad_month_year=",".join(form.fellowship_grad_month_year.data[:int(form.num_fellowships.data)]),
            bachelors=form.bachelors.data,
            bachelors_grad_month_year=form.bachelors_grad_month_year.data,
            msn=form.msn.data,
            msn_grad_month_year=form.msn_grad_month_year.data,
            dnp=form.dnp.data,
            dnp_grad_month_year=form.dnp_grad_month_year.data,
            additional_training=form.additional_training.data,
            sponsorship_needed=form.sponsorship_needed.data or False,
            malpractice_cases=json.dumps([
                {
                    'incident_year': case.form.incident_year.data,
                    'outcome': case.form.outcome.data,
                    'payout_amount': case.form.payout_amount.data or 0
                } for case in form.malpractice_cases.entries[:int(form.num_malpractice_cases.data)]
            ]),
            certification=form.certification.data,
            certification_specialty_area=form.certification_specialty_area.data,
            clinically_active=form.clinically_active.data,
            last_clinically_active=form.last_clinically_active.data if form.clinically_active.data == 'No' else None,
            emr=form.emr.data,
            languages=form.languages.data,
            states_licensed=",".join(form.states_licensed.data),
            states_willing_to_work=",".join(form.states_willing_to_work.data),
            salary_expectations=form.salary_expectations.data or 0.0,
            joined=datetime.utcnow()
        )

        db.session.add(new_doctor)
        db.session.commit()

        flash(f"Doctor profile and user '{username}' created successfully!", 'success')
        return redirect(url_for('doctors'))

    return render_template('add_doctor.html', form=form, zip=zip)




@app.route('/schedule_call', methods=['GET', 'POST'])
@login_required
def schedule_call():
    form = ScheduledCallForm()

    form.doctor_id.choices = [
        (doc.id, f"{doc.first_name} {doc.last_name} | {doc.email} | {doc.specialty}") 
        for doc in Doctor.query.order_by(Doctor.last_name).all()
    ]

    if form.validate_on_submit():
        dt = form.datetime.data  # <-- FIX HERE

        invite_status = "Pending" if request.form.get('send_invite') == 'yes' else "Accepted"

        call = ScheduledCall(
            doctor_id=form.doctor_id.data,
            scheduled_by_id=current_user.id,
            job_id=None,
            datetime=dt,
            reason=form.reason.data,
            invite_status=invite_status
        )
        db.session.add(call)
        db.session.commit()

        if invite_status == "Pending":
            doctor = Doctor.query.get(form.doctor_id.data)
            doctor_user = User.query.get(doctor.user_id)
            if doctor_user:
                message = Message(
                    sender_id=current_user.id,
                    recipient_id=doctor_user.id,
                    content=(
                        f"You have a new call invite scheduled on {dt.strftime('%Y-%m-%d %H:%M')} "
                        f"for reason: {form.reason.data}. Please accept or decline on your dashboard."
                    )
                )
                db.session.add(message)
                db.session.commit()
                flash('Invite sent to doctor successfully!', 'success')
        else:
            flash('Call scheduled successfully!', 'success')

        return redirect(url_for('home'))

    return render_template('schedule_call.html', form=form)



@app.route('/create_account')
def create_account():
    return render_template('create_account.html')


@app.route('/admin/dashboard')
@login_required
def admin_dashboard():
    if current_user.role != 'admin':
        flash('Unauthorized access!', 'danger')
        return redirect(url_for('home'))

    scheduled_calls = ScheduledCall.query.order_by(ScheduledCall.datetime).all()

    events = [
        {
            'id': call.id,
            'title': call.doctor.first_name + " " + call.doctor.last_name + " - " + (call.reason or "No Reason"),
            'start': call.datetime.strftime('%Y-%m-%dT%H:%M:%S'),
        } for call in scheduled_calls
    ]

    return render_template('index.html', events=events)


@app.route('/doctors')
def doctors():
    all_doctors = Doctor.query.all()
    return render_template('doctors.html', doctors=all_doctors)

@app.route('/edit_call/<int:call_id>', methods=['GET', 'POST'])
@login_required
def edit_call(call_id):
    scheduled_call = ScheduledCall.query.get_or_404(call_id)
    form = ScheduledCallForm(obj=scheduled_call)

    form.doctor_id.choices = [
        (doc.id, f"{doc.first_name} {doc.last_name} | {doc.email} | {doc.specialty}") 
        for doc in Doctor.query.order_by(Doctor.last_name).all()
    ]

    if form.validate_on_submit():
        # Check if datetime or doctor has changed
        original_datetime = scheduled_call.datetime
        original_doctor_id = scheduled_call.doctor_id

        scheduled_call.doctor_id = form.doctor_id.data
        scheduled_call.datetime = form.datetime.data if isinstance(form.datetime.data, datetime) else datetime.strptime(form.datetime.data, '%Y-%m-%dT%H:%M')
        scheduled_call.reason = form.reason.data
        
        db.session.commit()

        # Send notification to doctor's inbox if rescheduled
        if (scheduled_call.datetime != original_datetime or 
            scheduled_call.doctor_id != original_doctor_id):

            doctor_user = User.query.filter_by(doctor_id=scheduled_call.doctor_id).first()
            if doctor_user:
                message = Message(
                    sender_id=current_user.id,
                    recipient_id=doctor_user.id,
                    content=(
                        f"The scheduled meeting has been rescheduled to "
                        f"{scheduled_call.datetime.strftime('%Y-%m-%d %H:%M')} "
                        f"by {current_user.username}."
                    )
                )
                db.session.add(message)
                db.session.commit()

        flash('Scheduled call updated successfully!', 'success')
        return redirect(url_for('calls'))

    form.doctor_id.data = scheduled_call.doctor_id
    form.datetime.data = scheduled_call.datetime 
    form.reason.data = scheduled_call.reason

    return render_template('edit_call.html', form=form, call=scheduled_call)





#@app.route('/static/<path:filename>')
#def custom_static(filename):
    #full_path = r'C:\Users\Tubam\OneDrive\Desktop\static'
    #print(f"Serving file: {filename} from path: {full_path}")
    #return send_from_directory(full_path, filename)
@app.route('/')
def index():
    if current_user.is_authenticated:
        if current_user.role == 'doctor':
            return redirect(url_for('doctor_dashboard'))
        elif current_user.role == 'admin':
            return redirect(url_for('dashboard'))
        elif current_user.role == 'client':
            return redirect(url_for('client_dashboard'))
    return render_template('landing_page.html', current_year=datetime.now().year)

@app.route('/calls')
def calls():
    scheduled_calls = ScheduledCall.query.order_by(ScheduledCall.datetime).all()
    events = [
    {
        'id': call.id,
        'title': call.doctor.first_name + " " + call.doctor.last_name + " - " + (call.reason or "No Reason"),
        'start': call.datetime.strftime('%Y-%m-%dT%H:%M:%S'),
    } for call in scheduled_calls
    ]
    return render_template('calls.html', events=events)

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and user.check_password(form.password.data):
            login_user(user)
            flash('Logged in successfully!', 'success')
            
            # Redirect users immediately based on their role
            if user.role == 'doctor':
                return redirect(url_for('doctor_dashboard'))
            elif user.role == 'client':
                return redirect(url_for('client_dashboard'))
            elif user.role == 'admin':
                return redirect(url_for('dashboard'))
            else:
                flash('Role not recognized.', 'danger')
                return redirect(url_for('login'))

        flash('Invalid username or password.', 'danger')

    return render_template('login.html', form=form)


@app.route('/register_doctor', methods=['GET', 'POST'])
@login_required
def register_doctor():
    if current_user.role != 'admin':
        flash('Unauthorized access!', 'danger')
        return redirect(url_for('home'))

    form = DoctorRegistrationForm()
    if form.validate_on_submit():
        existing_user = User.query.filter(
            (User.username == form.username.data) | (User.email == form.email.data)
        ).first()

        if existing_user:
            flash('Username or email already exists for user login.', 'danger')
            return redirect(url_for('register_doctor'))

        # Check if doctor profile already exists with the email provided
        existing_doctor = Doctor.query.filter_by(email=form.email.data).first()

        if existing_doctor:
            # Create user linked to existing doctor profile
            new_user = User(
                username=form.username.data,
                email=form.email.data,
                password_hash=generate_password_hash(form.password.data),
                role='doctor'
            )
            db.session.add(new_user)
            db.session.commit()

            # Associate the existing doctor with the new user's ID
            existing_doctor.user_id = new_user.id
            db.session.commit()

            flash('Doctor login linked to existing profile!', 'success')
            return redirect(url_for('home'))

        # First create new user
        new_user = User(
            username=form.username.data,
            email=form.email.data,
            password_hash=generate_password_hash(form.password.data),
            role='doctor'
        )
        db.session.add(new_user)
        db.session.commit()

        # Then create new doctor profile associated to the new user
        new_doctor = Doctor(
            first_name='',
            last_name='',
            email=form.email.data,
            position='',
            specialty='',
            joined=datetime.utcnow(),
            user_id=new_user.id  # <-- Associate new doctor with new user
        )
        db.session.add(new_doctor)
        db.session.commit()

        # Finally, link newly created doctor profile back to user
        new_user.doctor_id = new_doctor.id
        db.session.commit()

        flash('New Doctor account created and linked!', 'success')
        return redirect(url_for('home'))

    return render_template('register_doctor.html', form=form)


@app.route('/scrape_jobs')
@login_required
def scrape_jobs():
    if current_user.role not in ['user', 'admin']:
        flash('Unauthorized access!', 'danger') 
        return redirect(url_for('home'))

    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from webdriver_manager.chrome import ChromeDriverManager
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, NoSuchElementException, ElementClickInterceptedException
    from bs4 import BeautifulSoup
    import time

    URL = 'https://www.doccafe.com/company/jobs'  

    options = webdriver.ChromeOptions()
    options.add_argument('--start-maximized')
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    driver.get(URL)

    input("✅ Please log into Doc Cafe now. Once you've logged in, press Enter to continue scraping...")

    jobs_added = 0

    try:
        while True:
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, "col-lg-8"))
            )

            soup = BeautifulSoup(driver.page_source, 'html.parser')
            job_elements = soup.find_all('div', class_='col-lg-8 col-sm-11 col-xs-10 no-padding-left')

            for job in job_elements:
                link_tag = job.find('h4').find('a', href=True)
                job_url = link_tag['href']
                job_title_tag = link_tag.find('span')
                job_title = job_title_tag.get_text(strip=True) if job_title_tag else 'N/A'

                # Open job detail page
                driver.execute_script("window.open(arguments[0]);", job_url)
                driver.switch_to.window(driver.window_handles[1])

                WebDriverWait(driver, 10).until(
                    EC.presence_of_element_located((By.CLASS_NAME, "job-summary"))
                )

                job_soup = BeautifulSoup(driver.page_source, 'html.parser')

                pay = job_soup.find('span', class_='job-salary-amount').get_text(strip=True) if job_soup.find('span', class_='job-salary-amount') else 'N/A'
                location = job_soup.find('span', class_='job-summary-location').get_text(strip=True) if job_soup.find('span', class_='job-summary-location') else 'N/A'
                description_tag = job_soup.find('div', class_='job-summary-box_field_6')
                description = description_tag.get_text(strip=True) if description_tag else 'N/A'

                # Save to database directly
                lat, lng = geocode_location(location)
                new_job = Job(
                    title=job_title,
                    location=location,
                    salary=pay,
                    description=description,
                    poster_id=current_user.id,
                    latitude=lat,
                    longitude=lng
                )
                db.session.add(new_job)
                db.session.commit()
                jobs_added += 1

                driver.close()
                driver.switch_to.window(driver.window_handles[0])

            try:
                next_button = driver.find_element(By.XPATH, '//a[@rel="next"]')
                driver.execute_script("arguments[0].click();", next_button)
                time.sleep(3)
            except (NoSuchElementException, ElementClickInterceptedException):
                flash("No more pages or unable to click next button.", "info")
                break

    except TimeoutException:
        flash("Timed out waiting for page to load.", "danger")
    finally:
        driver.quit()

    flash(f"{jobs_added} job postings successfully added!", "success")
    return redirect(url_for('doctor_jobs'))


@app.route('/post_job', methods=['GET', 'POST'])
@login_required
def post_job():
    if current_user.role not in ['client', 'admin']:
        flash('Only clients and admins can post jobs!', 'danger')
        return redirect(url_for('home'))

    form = JobForm()
    if form.validate_on_submit():
        # Geocode location before creating the Job
        lat, lng = geocode_location(form.location.data)
        job = Job(
            title=form.title.data,
            location=form.location.data,
            salary=form.salary.data,
            description=form.description.data,
            poster_id=current_user.id,
            latitude=lat,
            longitude=lng
        )
        db.session.add(job)
        db.session.commit()
        flash('Job posted successfully!', 'success')

        if current_user.role == 'admin':
            return redirect(url_for('home'))
        elif current_user.role == 'client':
            return redirect(url_for('client_dashboard'))

    return render_template('post_job.html', form=form)

@app.route('/doctor/ai_search_jobs', methods=['POST'])
@login_required
def doctor_ai_search_jobs():
    if current_user.role != 'doctor':
        return jsonify({'error': 'Unauthorized'}), 401

    lifestyle = request.form.get('lifestyle', '')
    wants = request.form.get('wants', '')
    location = request.form.get('location', '')
    jobs = Job.query.order_by(Job.id.desc()).all()

    jobs_payload = [
        {
            "id": job.id,
            "title": job.title,
            "location": job.location,
            "salary": job.salary,
            "description": job.description
        } for job in jobs
    ]

    prompt = f"""
You are a professional AI assistant that matches doctors with jobs.
You must only reference the information provided in the job listings and the doctor's preferences below—do not invent or assume lifestyle, city perks, or job benefits unless explicitly present in the job description.
Your output must always use consistent, professional HTML, styled in a blue color scheme (#0066cc for headers and buttons), and avoid any images.  
Show four sections:  
1. The Best Overall Fit (the single job that best matches ALL criteria together).  
2. The Best Fit for Location (the job most aligned with the doctor's location preference, even if it's also the overall fit).
3. The Best Fit for Lifestyle (the job most aligned with the lifestyle preference).
4. The Best Fit for Job-Specific Wants (the job most aligned with the job-specific wants).

For each section:
- Use a prominent blue header (e.g., `<h2 style="color:#0066cc;">Best Overall Fit</h2>`)
- In a card-style `<div>` (light blue background: #eaf2fb, rounded corners, subtle blue border), show:
    - Job Title (big, bold, blue: #0066cc)
    - Location
    - Salary (if available)
    - Professional, vivid summary explaining why this job fits the criteria, referencing the doctor's preferences.
    - A large blue "View Job" button: `<a href="/doctor/job/{{job_id}}" style="background:#0066cc;color:white;padding:10px 28px;font-size:1.1em;border-radius:7px;display:inline-block;text-decoration:none;margin-top:10px;">View Job</a>`
    - Below the card, show a short "Why This Job?" blurb in italic, describing why this was selected for that category.
- Do not use code blocks.
- Do not include any images, icons, or emojis.
- If a job is selected in multiple categories, show it again in those sections (repeat is OK).
- Only pick from the jobs provided below.

Doctor's preferences:
- Lifestyle: {lifestyle}
- Job-specific wants: {wants}
- Location: {location}

Job list (each includes id, title, location, salary, description):
{json.dumps(jobs_payload, indent=2)}

Your response must be fully-rendered HTML, ready to be dropped into a modal. 
The main title should be <h1 style="color:#0066cc;font-size:2.3em;margin-bottom:0.3em;">Exciting Job Opportunities Tailored For You!</h1>
Do not output any <img> tags or links to images. Only output the requested job sections, each with consistent blue theme. 
    """

    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4.1-nano",
        messages=[
            {"role": "system", "content": "You are a professional and creative medical job match assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=2200,
        temperature=0.5,  # moderate creativity, more consistent
    )
    gpt_html = response.choices[0].message.content

    return jsonify({'html': gpt_html})

@app.route('/doctor/jobs')
@login_required
def doctor_jobs():
    if current_user.role != 'doctor':
        flash('Unauthorized access.', 'danger')
        return redirect(url_for('home'))

    keyword = request.args.get('keyword', '').lower()
    location = request.args.get('location', '').lower()

    jobs_query = Job.query

    if keyword:
        jobs_query = jobs_query.filter(Job.title.ilike(f"%{keyword}%") | Job.description.ilike(f"%{keyword}%"))
    if location:
        jobs_query = jobs_query.filter(Job.location.ilike(f"%{location}%"))



    jobs = jobs_query.order_by(Job.id.desc()).all()
    marker_groups = defaultdict(list)
    for job in jobs:
        if job.latitude and job.longitude:
            key = (round(job.latitude, 5), round(job.longitude, 5))  # rounding avoids float mismatch
            marker_groups[key].append(job)

    job_markers = []
    for (lat, lng), joblist in marker_groups.items():
        job_entries = []
        for job in joblist:
            job_entries.append({
                "id": job.id,
                "title": job.title,
                "location": job.location,
                "salary": job.salary,
            })
        job_markers.append({
            "lat": lat,
            "lng": lng,
            "jobs": job_entries
        })
    return render_template('doctor_jobs.html', jobs=jobs, job_markers=job_markers, keyword=keyword, location=location)

    

@app.route('/doctor/job/<int:job_id>', methods=['GET', 'POST'])
@login_required
def view_job(job_id):
    if current_user.role != 'doctor':
        flash('Access denied.', 'danger')
        return redirect(url_for('home'))

    job = Job.query.get_or_404(job_id)

    # Check if doctor already expressed interest
    already_interested = Message.query.filter_by(
        sender_id=current_user.id,
        job_id=job.id,
        message_type='interest'
    ).first()

    if request.method == 'POST':
        if already_interested:
            flash('You have already expressed interest in this job.', 'info')
        else:
            recipient_user = job.poster
            message = Message(
                sender_id=current_user.id,
                recipient_id=recipient_user.id,
                job_id=job.id,
                doctor_id=current_user.doctor.id,
                content=f"Dr. {current_user.doctor.first_name} {current_user.doctor.last_name} expressed interest in your job '{job.title}'.",
                message_type='interest'
            )
            db.session.add(message)
            db.session.commit()
            flash('Your interest has been sent to the client.', 'success')
        return redirect(url_for('doctor_jobs'))

    return render_template('view_job.html', job=job, already_interested=already_interested)



@app.route('/doctor/dashboard')
@login_required
def doctor_dashboard():
    doctor = current_user.doctor

    scheduled_calls = ScheduledCall.query.filter_by(doctor_id=doctor.id).all()
    pending_invites = ScheduledCall.query.filter_by(doctor_id=doctor.id, invite_status='Pending').all()

    events = []
    for call in scheduled_calls:
        # Determine the color and status clearly
        if call.canceled:
            color, status = '#ff4d4d', 'Canceled'
        elif call.reschedule_requested:
            color, status = '#3788d8', 'Reschedule Requested'
        elif call.invite_status.lower() == 'pending':
            color, status = '#ffc107', 'Pending Invite'
        elif call.invite_status.lower() == 'accepted':
            color, status = '#28a745', 'Accepted'
        else:
            color, status = '#6c757d', 'Scheduled'

        events.append({
            'id': call.id,
            'title': f"Call with {call.scheduled_by.username}",
            'start': call.datetime.isoformat(),
            'color': color,
            'status': status
        })

    return render_template('doctor_dashboard.html', doctor=doctor, events=events, pending_invites=pending_invites)


@app.route('/handle_invite/<int:call_id>', methods=['POST'])
@login_required
def handle_invite(call_id):
    scheduled_call = ScheduledCall.query.get_or_404(call_id)

    if current_user.doctor.id != scheduled_call.doctor_id:
        flash('Unauthorized.', 'danger')
        return redirect(url_for('doctor_dashboard'))

    action = request.form.get('action')
    if action == 'accept':
        scheduled_call.invite_status = 'accepted'
        flash('Invite accepted.', 'success')

        # Notify client of acceptance
        content = f"Dr. {current_user.doctor.first_name} {current_user.doctor.last_name} accepted your call invite."
    elif action == 'decline':
        scheduled_call.invite_status = 'declined'
        flash('Invite declined.', 'info')

        # Notify client of decline
        content = f"Dr. {current_user.doctor.first_name} {current_user.doctor.last_name} declined your call invite."
    else:
        flash('Invalid action.', 'danger')
        return redirect(url_for('doctor_dashboard'))

    # Send notification message back to client
    notification = Message(
        sender_id=current_user.id,
        recipient_id=scheduled_call.scheduled_by_id,
        doctor_id=current_user.doctor.id,
        content=content,
        message_type='invite_response'
    )

    db.session.add(notification)
    db.session.commit()

    return redirect(url_for('doctor_dashboard'))


@app.route('/doctor/handle_invite/<int:call_id>', methods=['POST'])
@login_required
def doctor_handle_invite(call_id):
    scheduled_call = ScheduledCall.query.get_or_404(call_id)

    action = request.form.get('action')

    if action == 'accept':
        scheduled_call.invite_status = "Accepted"
        flash('Invite accepted!', 'success')

        notification = Message(
            sender_id=current_user.id,
            recipient_id=scheduled_call.scheduled_by_id,
            content=f"Dr. {current_user.doctor.first_name} {current_user.doctor.last_name} accepted your meeting invite scheduled on {scheduled_call.datetime.strftime('%Y-%m-%d %H:%M')}."
        )

    elif action == 'decline':
        scheduled_call.invite_status = "Declined"
        flash('Invite declined!', 'danger')

        notification = Message(
            sender_id=current_user.id,
            recipient_id=scheduled_call.scheduled_by_id,
            content=f"Dr. {current_user.doctor.first_name} {current_user.doctor.last_name} declined your meeting invite scheduled on {scheduled_call.datetime.strftime('%Y-%m-%d %H:%M')}."
        )

    db.session.add(notification)
    db.session.commit()

    return redirect(url_for('doctor_dashboard'))

@app.route('/admin/analytics')
@login_required
def admin_analytics():
    if current_user.role != 'admin':
        flash('Unauthorized access!', 'danger')
        return redirect(url_for('home'))

    jobs = Job.query.order_by(Job.id.desc()).all()

    job_data = []
    for job in jobs:
        messages = Message.query.filter_by(job_id=job.id, message_type='interest').all()

        # Count interest per day
        interest_by_day = defaultdict(int)
        interested_doctors = []
        for msg in messages:
            date_str = msg.timestamp.strftime("%Y-%m-%d")
            interest_by_day[date_str] += 1

            if msg.doctor:
                interested_doctors.append({
                    'id': msg.doctor.id,
                    'name': f"{msg.doctor.first_name} {msg.doctor.last_name}",
                    'email': msg.doctor.email
                })

        job_data.append({
            'id': job.id,
            'title': job.title,
            'location': job.location,
            'salary': job.salary,
            'description': job.description,
            'interest_count': len(messages),
            'interested_doctors': interested_doctors,
            'interest_by_day': interest_by_day,
            'client_name': job.poster.username if job.poster else "Unknown",
            'client_email': job.poster.email if job.poster else "",
        })

    return render_template("admin_analytics.html", job_data=job_data)

@app.route('/doctor/handle_reschedule/<int:call_id>', methods=['GET', 'POST'])
@login_required
def doctor_handle_reschedule(call_id):
    scheduled_call = ScheduledCall.query.get_or_404(call_id)

    if current_user.id != scheduled_call.doctor.user.id:
        flash('Unauthorized', 'danger')
        return redirect(url_for('doctor_dashboard'))

    if request.method == 'POST':
        action = request.form.get('action')

        if action == 'accept':
            scheduled_call.datetime = scheduled_call.reschedule_datetime
            scheduled_call.reschedule_requested = False
            scheduled_call.reschedule_note = None
            scheduled_call.reschedule_datetime = None
            db.session.commit()
            flash('Reschedule accepted.', 'success')

        elif action == 'reject':
            scheduled_call.reschedule_requested = False
            scheduled_call.reschedule_note = None
            scheduled_call.reschedule_datetime = None
            db.session.commit()
            flash('Reschedule request rejected.', 'warning')

        return redirect(url_for('doctor_dashboard'))

    return render_template('handle_reschedule.html', call=scheduled_call)

@app.route('/register_client', methods=['GET', 'POST'])
@login_required
def register_client():
    if current_user.role != 'admin':
        flash('Unauthorized access!', 'danger')
        return redirect(url_for('home'))

    form = ClientRegistrationForm()
    if form.validate_on_submit():
        existing_user = User.query.filter(
            (User.username == form.username.data) | (User.email == form.email.data)
        ).first()

        if existing_user:
            flash('Username or email already exists.', 'danger')
            return redirect(url_for('register_client'))

        new_client = User(
            username=form.username.data,
            email=form.email.data,
            password_hash=generate_password_hash(form.password.data),
            role='client'
        )
        db.session.add(new_client)
        db.session.commit()

        flash('New client account created successfully!', 'success')
        return redirect(url_for('home'))

    return render_template('register_client.html', form=form)

@app.route('/doctor/job/<int:job_id>/express_interest', methods=['POST'])
@login_required
def express_interest(job_id):
    job = Job.query.get_or_404(job_id)

    if current_user.role != 'doctor':
        flash('Unauthorized', 'danger')
        return redirect(url_for('doctor_jobs'))

    recipient_user = job.poster

    message = Message(
        sender_id=current_user.id,
        recipient_id=recipient_user.id,
        job_id=job.id,
        doctor_id=current_user.doctor.id,
        content=f"{current_user.doctor.first_name} {current_user.doctor.last_name} expressed interest in your job: '{job.title}'.",
        message_type='interest'  # <-- clearly marked interest message
    )

    db.session.add(message)
    db.session.commit()

    flash('Interest sent to client!', 'success')
    return redirect(url_for('doctor_jobs'))

@app.route('/send_invite/<int:doctor_id>/<int:job_id>', methods=['GET', 'POST'])
@login_required
def send_invite(doctor_id, job_id):
    if current_user.role not in ['client', 'admin']:
        flash('Unauthorized access.', 'danger')
        return redirect(url_for('home'))

    doctor = Doctor.query.get_or_404(doctor_id)
    doctor_user = User.query.filter_by(id=doctor.user_id).first_or_404()
    job = Job.query.get_or_404(job_id)

    form = ScheduledCallForm()
    form.doctor_id.choices = [(doctor.id, f"{doctor.first_name} {doctor.last_name} | {doctor.email}")]

    if form.validate_on_submit():
        scheduled_call = ScheduledCall(
            doctor_id=doctor.id,
            scheduled_by_id=current_user.id,
            job_id=job.id,
            datetime=form.datetime.data,
            reason=form.reason.data,
            invite_status='pending'
        )

        db.session.add(scheduled_call)
        db.session.commit()

        # Send invite message to doctor
        message = Message(
            sender_id=current_user.id,
            recipient_id=doctor_user.id,
            job_id=job_id,
            doctor_id=doctor.id,
            content=f"You have a call invite scheduled by {current_user.username} on {form.datetime.data}.",
            message_type='invite'
        )
        db.session.add(message)
        db.session.commit()

        flash('Invite sent to doctor!', 'success')

        # Redirect to appropriate dashboard
        if current_user.role == 'admin':
            return redirect(url_for('admin_dashboard'))
        else:
            return redirect(url_for('client_dashboard'))

    return render_template('schedule_call.html', form=form, job=job, doctor=doctor)


@app.route('/edit_job/<int:job_id>', methods=['GET', 'POST'])
@login_required
def edit_job(job_id):
    job = Job.query.get_or_404(job_id)

    if current_user.role != 'client' or job.poster_id != current_user.id:
        flash('Unauthorized access!', 'danger')
        return redirect(url_for('home'))

    form = JobForm(obj=job)

    if form.validate_on_submit():
        job.title = form.title.data
        job.location = form.location.data
        job.salary = form.salary.data
        job.description = form.description.data

        db.session.commit()
        flash('Job updated successfully!', 'success')
        return redirect(url_for('client_my_jobs'))

    return render_template('edit_job.html', form=form, job=job)


# ✅ Full route: doctor_edit_profile
@app.route('/doctor/edit_profile', methods=['GET', 'POST'])
@login_required
def doctor_edit_profile():
    doctor = current_user.doctor
    form = DoctorForm()

    if request.method == 'POST':
        if form.validate_on_submit():
            try:
                existing_doctor = Doctor.query.filter(
                    Doctor.email == form.email.data, Doctor.id != doctor.id
                ).first()
                if existing_doctor:
                    flash('Another doctor with this email already exists.', 'danger')
                    return redirect(url_for('edit_doctor', doctor_id=doctor.id))

                # ONLY handle image update if cropper input is submitted
                cropped_data = request.form.get('cropped_image_data')
                if cropped_data:
                    import base64
                    from PIL import Image
                    from io import BytesIO

                    header, encoded = cropped_data.split(",", 1)
                    binary_data = base64.b64decode(encoded)
                    img = Image.open(BytesIO(binary_data))

                    cropped_filename = f"cropped_{datetime.now().strftime('%Y%m%d%H%M%S')}.png"
                    cropped_path = os.path.join(app.static_folder, 'upload', cropped_filename)
                    img.save(cropped_path)
                    doctor.profile_picture = f"upload/{cropped_filename}"

                doctor.position = form.position.data
                doctor.specialty = form.specialty.data
                doctor.subspecialty = form.subspecialty.data
                doctor.first_name = form.first_name.data
                doctor.last_name = form.last_name.data
                doctor.email = form.email.data
                doctor.phone = form.phone.data
                doctor.alt_phone = form.alt_phone.data
                doctor.city_of_residence = form.city_of_residence.data
                doctor.medical_school = form.medical_school.data
                doctor.med_grad_month_year = form.med_grad_month_year.data
                doctor.residency = form.residency.data
                doctor.residency_grad_month_year = form.residency_grad_month_year.data

                num_fellowships = int(form.num_fellowships.data)
                doctor.fellowship = ",".join(form.fellowship.data[:num_fellowships])
                doctor.fellowship_grad_month_year = ",".join(form.fellowship_grad_month_year.data[:num_fellowships])

                doctor.bachelors = form.bachelors.data
                doctor.bachelors_grad_month_year = form.bachelors_grad_month_year.data
                doctor.msn = form.msn.data
                doctor.msn_grad_month_year = form.msn_grad_month_year.data
                doctor.dnp = form.dnp.data
                doctor.dnp_grad_month_year = form.dnp_grad_month_year.data
                doctor.additional_training = form.additional_training.data
                doctor.sponsorship_needed = form.sponsorship_needed.data or False

                num_cases = int(form.num_malpractice_cases.data)
                malpractice_data = []
                for case_form in form.malpractice_cases.entries[:num_cases]:
                    malpractice_data.append({
                        'incident_year': case_form.form.incident_year.data,
                        'outcome': case_form.form.outcome.data,
                        'payout_amount': case_form.form.payout_amount.data or 0
                    })
                doctor.malpractice_cases = json.dumps(malpractice_data)

                doctor.certification = form.certification.data
                doctor.certification_specialty_area = form.certification_specialty_area.data
                doctor.clinically_active = form.clinically_active.data
                doctor.last_clinically_active = form.last_clinically_active.data if form.clinically_active.data == 'No' else None
                doctor.emr = form.emr.data
                doctor.languages = form.languages.data
                doctor.states_licensed = ",".join(form.states_licensed.data)
                doctor.states_willing_to_work = ",".join(form.states_willing_to_work.data)
                doctor.salary_expectations = form.salary_expectations.data or 0.0

                db.session.commit()
                flash('Doctor updated successfully!', 'success')
                return redirect(url_for('doctor_dashboard'))

            except Exception as e:
                db.session.rollback()
                flash(f"An error occurred: {str(e)}", "danger")

    if request.method == 'GET':
        form.position.data = doctor.position
        form.specialty.data = doctor.specialty
        form.subspecialty.data = doctor.subspecialty
        form.first_name.data = doctor.first_name
        form.last_name.data = doctor.last_name
        form.email.data = doctor.email
        form.phone.data = doctor.phone
        form.alt_phone.data = doctor.alt_phone
        form.city_of_residence.data = doctor.city_of_residence
        form.medical_school.data = doctor.medical_school
        form.med_grad_month_year.data = doctor.med_grad_month_year
        form.residency.data = doctor.residency
        form.residency_grad_month_year.data = doctor.residency_grad_month_year

        fellowships = doctor.fellowship.split(',') if doctor.fellowship else []
        fellowship_dates = doctor.fellowship_grad_month_year.split(',') if doctor.fellowship_grad_month_year else []
        form.num_fellowships.data = str(len(fellowships))
        form.fellowship.entries.clear()
        form.fellowship_grad_month_year.entries.clear()
        for f, d in zip(fellowships, fellowship_dates):
            form.fellowship.append_entry(f)
            form.fellowship_grad_month_year.append_entry(d)
        while len(form.fellowship.entries) < int(form.fellowship.max_entries):
            form.fellowship.append_entry()
            form.fellowship_grad_month_year.append_entry()

        malpractice_cases = json.loads(doctor.malpractice_cases or '[]')
        form.num_malpractice_cases.data = str(len(malpractice_cases))
        form.malpractice_cases.entries.clear()
        for case in malpractice_cases:
            entry = form.malpractice_cases.append_entry()
            entry.incident_year.data = case.get('incident_year', '')
            entry.outcome.data = case.get('outcome', '')
            entry.payout_amount.data = case.get('payout_amount', 0.0)
        while len(form.malpractice_cases.entries) < int(form.malpractice_cases.max_entries):
            form.malpractice_cases.append_entry()

        form.bachelors.data = doctor.bachelors
        form.bachelors_grad_month_year.data = doctor.bachelors_grad_month_year
        form.msn.data = doctor.msn
        form.msn_grad_month_year.data = doctor.msn_grad_month_year
        form.dnp.data = doctor.dnp
        form.dnp_grad_month_year.data = doctor.dnp_grad_month_year
        form.additional_training.data = doctor.additional_training
        form.sponsorship_needed.data = doctor.sponsorship_needed
        form.certification.data = doctor.certification
        form.certification_specialty_area.data = doctor.certification_specialty_area
        form.clinically_active.data = doctor.clinically_active
        form.last_clinically_active.data = doctor.last_clinically_active
        form.emr.data = doctor.emr
        form.languages.data = doctor.languages
        form.states_licensed.data = doctor.states_licensed.split(',') if doctor.states_licensed else []
        form.states_willing_to_work.data = doctor.states_willing_to_work.split(',') if doctor.states_willing_to_work else []
        form.salary_expectations.data = doctor.salary_expectations

    return render_template('doctor_edit_profile.html', form=form, doctor=doctor, zip=zip)





@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('Logged out successfully.', 'success')
    return redirect(url_for('login'))

@app.route('/client/inbox')
@login_required
def client_inbox():
    if current_user.role != 'client':
        flash('Access denied.', 'danger')
        return redirect(url_for('home'))

    messages = Message.query.filter_by(recipient_id=current_user.id).order_by(Message.timestamp.desc()).all()

    unread_messages = [msg for msg in messages if not msg.read]
    for msg in unread_messages:
        msg.read = True
    db.session.commit()

    return render_template('inbox.html', messages=messages, title="Client Inbox")

@app.route('/register', methods=['GET', 'POST'])
@login_required  # Only logged-in users can create new accounts
def register():
    form = RegisterForm()
    if form.validate_on_submit():
        existing_user = User.query.filter_by(username=form.username.data).first()
        if existing_user:
            flash('Username already exists.', 'danger')
        else:
            new_user = User(username=form.username.data)
            new_user.set_password(form.password.data)
            db.session.add(new_user)
            db.session.commit()
            flash('New user registered successfully!', 'success')
            return redirect(url_for('home'))
    return render_template('register.html', form=form)

@app.route('/test_db')
def test_db():
    try:
        db.session.execute("SELECT 1")
        return "Connection successful!"
    except Exception as e:
        return f"Connection failed: {e}"

@app.route('/client/dashboard')
@login_required
def client_dashboard():
    scheduled_calls = ScheduledCall.query.filter_by(scheduled_by_id=current_user.id).all()
    reschedule_requests = ScheduledCall.query.filter_by(
        scheduled_by_id=current_user.id, reschedule_requested=True
    ).all()

    events = []
    for call in scheduled_calls:
        if call.canceled:
            color, status = '#dc3545', 'Canceled'
        elif call.reschedule_requested:
            color, status = '#17a2b8', 'Reschedule Requested'
        elif call.invite_status == 'Pending':
            color, status = '#ffc107', 'Pending Invite'
        else:
            color, status = '#28a745', 'Accepted'

        events.append({
            'id': call.id,
            'title': f"Call with Dr. {call.doctor.first_name} {call.doctor.last_name}",
            'start': call.datetime.isoformat(),
            'color': color,
            'status': status,
        })

    return render_template(
        'client_dashboard.html',
        events=events,
        reschedule_requests=reschedule_requests
    )



@app.route('/dashboard')
@login_required
def dashboard():
    if current_user.role == 'admin':
        return redirect(url_for('admin_dashboard'))
    elif current_user.role == 'client':
        return redirect(url_for('client_dashboard'))
    elif current_user.role == 'doctor':
        return redirect(url_for('doctor_dashboard'))
    else:
        flash('Role not recognized.', 'danger')
        return redirect(url_for('login'))

@app.route('/doctor/call/<int:call_id>', methods=['GET', 'POST'])
@login_required
def doctor_call_details(call_id):
    scheduled_call = ScheduledCall.query.get_or_404(call_id)

    if current_user.role != 'doctor' or scheduled_call.doctor_id != current_user.doctor.id:
        flash('Unauthorized', 'danger')
        return redirect(url_for('doctor_dashboard'))

    if request.method == 'POST':
        action = request.form.get('action')

        if action == 'cancel':
            scheduled_call.canceled = True
            db.session.commit()
            flash('Meeting canceled.', 'success')
            return redirect(url_for('doctor_dashboard'))

        elif action == 'reschedule':
            new_datetime = request.form.get('reschedule_datetime')
            note = request.form.get('reschedule_note')

            scheduled_call.reschedule_requested = True
            scheduled_call.reschedule_note = note
            scheduled_call.reschedule_datetime = datetime.strptime(new_datetime, '%Y-%m-%dT%H:%M')
            db.session.commit()

            # Put the notification code HERE
            message = Message(
                sender_id=current_user.id,
                recipient_id=scheduled_call.scheduled_by_id,
                content=f"Reschedule requested for call on {scheduled_call.datetime.strftime('%Y-%m-%d %H:%M')} to {scheduled_call.reschedule_datetime.strftime('%Y-%m-%d %H:%M')}. Reason: {scheduled_call.reschedule_note}"
            )
            db.session.add(message)
            db.session.commit()

            flash('Reschedule requested sent.', 'success')
            return redirect(url_for('doctor_dashboard'))

    return render_template('doctor_call_details.html', call=scheduled_call)

@app.route('/client/my_jobs')
@login_required
def client_my_jobs():
    if current_user.role != 'client':
        flash('Unauthorized access!', 'danger')
        return redirect(url_for('home'))

    keyword = request.args.get('keyword', '').lower()
    location = request.args.get('location', '').lower()

    jobs_query = Job.query.filter_by(poster_id=current_user.id)

    if keyword:
        jobs_query = jobs_query.filter(Job.title.ilike(f"%{keyword}%") | Job.description.ilike(f"%{keyword}%"))
    if location:
        jobs_query = jobs_query.filter(Job.location.ilike(f"%{location}%"))

    jobs = jobs_query.order_by(Job.id.desc()).all()

    return render_template('client_my_jobs.html', jobs=jobs, keyword=keyword, location=location)



@app.route('/doctor/<int:doctor_id>')
def doctor_profile(doctor_id):
    doctor = Doctor.query.get_or_404(doctor_id)
    malpractice_cases = json.loads(doctor.malpractice_cases or '[]')
    return render_template('doctor_profile.html', doctor=doctor, malpractice_cases=malpractice_cases)

@app.route('/client/handle_reschedule/<int:call_id>', methods=['POST'])
@login_required
def client_handle_reschedule(call_id):
    scheduled_call = ScheduledCall.query.get_or_404(call_id)
    action = request.form.get('action')
    client_note = request.form.get('client_note', '')

    if action == 'accept':
        scheduled_call.datetime = scheduled_call.reschedule_datetime
        scheduled_call.reschedule_requested = False
        scheduled_call.reschedule_note = None
        scheduled_call.reschedule_datetime = None
        content = (f"Your reschedule request for "
                   f"{scheduled_call.datetime.strftime('%Y-%m-%d %H:%M')} has been accepted. {client_note}")
    elif action == 'decline':
        scheduled_call.reschedule_requested = False
        declined_datetime = scheduled_call.reschedule_datetime
        scheduled_call.reschedule_datetime = None
        scheduled_call.reschedule_note = None
        content = (f"Your reschedule request for "
                   f"{declined_datetime.strftime('%Y-%m-%d %H:%M')} has been declined. {client_note}")

    db.session.commit()

    message = Message(
        sender_id=current_user.id,
        recipient_id=scheduled_call.doctor.user_id,  # Corrected this line
        content=content,
        timestamp=datetime.utcnow()
    )

    db.session.add(message)
    db.session.commit()

    flash('Reschedule handled successfully.', 'success')
    return redirect(url_for('client_dashboard'))


@app.route('/client/analytics/download/<int:job_id>')
@login_required
def download_job_applicants(job_id):
    job = Job.query.get_or_404(job_id)
    # Allow if admin OR the client who posted the job
    if current_user.role not in ['client', 'admin']:
        flash('Unauthorized access!', 'danger')
        return redirect(url_for('home'))
    if current_user.role == 'client' and job.poster_id != current_user.id:
        flash('Not authorized for this job.', 'danger')
        return redirect(url_for('client_analytics'))

    # Get all doctors who expressed interest (via Message table)
    messages = Message.query.filter_by(job_id=job.id, message_type='interest').all()
    doctor_ids = {msg.doctor_id for msg in messages if msg.doctor_id}

    # Get Doctor objects
    doctors = Doctor.query.filter(Doctor.id.in_(doctor_ids)).all()

    # Prepare data for Excel (all fields in Doctor model)
    rows = []
    for doc in doctors:
        row = {
            "First Name": doc.first_name,
            "Last Name": doc.last_name,
            "Position": doc.position,
            "Specialty": doc.specialty,
            "Subspecialty": doc.subspecialty,
            "Email": doc.email,
            "Phone": doc.phone,
            "Alt Phone": doc.alt_phone,
            "City of Residence": doc.city_of_residence,
            "Medical School": doc.medical_school,
            "Med Grad Month/Year": doc.med_grad_month_year,
            "Residency": doc.residency,
            "Residency Grad Month/Year": doc.residency_grad_month_year,
            "Fellowship(s)": doc.fellowship,
            "Fellowship Grad Month/Year": doc.fellowship_grad_month_year,
            "Bachelors": doc.bachelors,
            "Bachelors Grad Month/Year": doc.bachelors_grad_month_year,
            "MSN": doc.msn,
            "MSN Grad Month/Year": doc.msn_grad_month_year,
            "DNP": doc.dnp,
            "DNP Grad Month/Year": doc.dnp_grad_month_year,
            "Additional Training": doc.additional_training,
            "Sponsorship Needed": "Yes" if doc.sponsorship_needed else "No",
            "Certification": doc.certification,
            "Certification Specialty": doc.certification_specialty_area,
            "Clinically Active": doc.clinically_active,
            "Last Clinically Active": doc.last_clinically_active,
            "EMR": doc.emr,
            "Languages": doc.languages,
            "States Licensed": doc.states_licensed,
            "States Willing to Work": doc.states_willing_to_work,
            "Salary Expectations": doc.salary_expectations,
            "Joined": doc.joined.strftime('%Y-%m-%d') if doc.joined else "",
            # You can add more fields here if needed!
        }
        # Optionally, include Malpractice cases (as JSON string)
        row["Malpractice Cases"] = doc.malpractice_cases
        rows.append(row)

    # Generate Excel
    df = pd.DataFrame(rows)
    output = BytesIO()
    df.to_excel(output, index=False)
    output.seek(0)

    filename = f"Applicants_{job.title.replace(' ','_')}.xlsx"
    return send_file(
        output,
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        download_name=filename,
        as_attachment=True
    )

@app.route('/edit_doctor/<int:doctor_id>', methods=['GET', 'POST'])
@login_required
def edit_doctor(doctor_id):
    doctor = Doctor.query.get_or_404(doctor_id)
    form = DoctorForm()

    if form.validate_on_submit():
        try:
            existing_doctor = Doctor.query.filter(
                Doctor.email == form.email.data, Doctor.id != doctor_id
            ).first()
            if existing_doctor:
                flash('Another doctor with this email already exists.', 'danger')
                return redirect(url_for('edit_doctor', doctor_id=doctor_id))

            doctor.position = form.position.data
            doctor.specialty = form.specialty.data
            doctor.subspecialty = form.subspecialty.data
            doctor.first_name = form.first_name.data
            doctor.last_name = form.last_name.data
            doctor.email = form.email.data
            doctor.phone = form.phone.data
            doctor.alt_phone = form.alt_phone.data
            doctor.city_of_residence = form.city_of_residence.data
            doctor.medical_school = form.medical_school.data
            doctor.med_grad_month_year = form.med_grad_month_year.data
            doctor.residency = form.residency.data
            doctor.residency_grad_month_year = form.residency_grad_month_year.data

            # Handle Fellowships dynamically
            num_fellowships = int(form.num_fellowships.data)
            doctor.fellowship = ",".join(form.fellowship.data[:num_fellowships])
            doctor.fellowship_grad_month_year = ",".join(form.fellowship_grad_month_year.data[:num_fellowships])

            doctor.bachelors = form.bachelors.data
            doctor.bachelors_grad_month_year = form.bachelors_grad_month_year.data
            doctor.msn = form.msn.data
            doctor.msn_grad_month_year = form.msn_grad_month_year.data
            doctor.dnp = form.dnp.data
            doctor.dnp_grad_month_year = form.dnp_grad_month_year.data
            doctor.additional_training = form.additional_training.data
            doctor.sponsorship_needed = form.sponsorship_needed.data or False

            # Handle Malpractice Cases dynamically
            num_cases = int(form.num_malpractice_cases.data)
            malpractice_data = []
            for case_form in form.malpractice_cases.entries[:num_cases]:
                malpractice_data.append({
                    'incident_year': case_form.form.incident_year.data,
                    'outcome': case_form.form.outcome.data,
                    'payout_amount': case_form.form.payout_amount.data or 0
                })
            doctor.malpractice_cases = json.dumps(malpractice_data)

            doctor.certification = form.certification.data
            doctor.certification_specialty_area = form.certification_specialty_area.data
            doctor.clinically_active = form.clinically_active.data

            if form.clinically_active.data == 'No':
                doctor.last_clinically_active = form.last_clinically_active.data
            else:
                doctor.last_clinically_active = None
            doctor.emr = form.emr.data
            doctor.languages = form.languages.data
            doctor.states_licensed = ",".join(form.states_licensed.data)
            doctor.states_willing_to_work = ",".join(form.states_willing_to_work.data)
            doctor.salary_expectations = form.salary_expectations.data or 0.0

            db.session.commit()
            flash('Doctor updated successfully!', 'success')
            return redirect(url_for('doctor_profile', doctor_id=doctor.id))

        except Exception as e:
            db.session.rollback()
            flash(f"An error occurred: {str(e)}", "danger")

    if request.method == 'GET':
    # Manually set simple fields
        form.position.data = doctor.position
        form.specialty.data = doctor.specialty
        form.subspecialty.data = doctor.subspecialty
        form.first_name.data = doctor.first_name
        form.last_name.data = doctor.last_name
        form.email.data = doctor.email
        form.phone.data = doctor.phone
        form.alt_phone.data = doctor.alt_phone
        form.city_of_residence.data = doctor.city_of_residence
        form.medical_school.data = doctor.medical_school
        form.med_grad_month_year.data = doctor.med_grad_month_year
        form.residency.data = doctor.residency
        form.residency_grad_month_year.data = doctor.residency_grad_month_year

        # Fellowships (with limit)
        form.fellowship.entries.clear()
        form.fellowship_grad_month_year.entries.clear()
        fellowships = doctor.fellowship.split(',') if doctor.fellowship else []
        fellowship_dates = doctor.fellowship_grad_month_year.split(',') if doctor.fellowship_grad_month_year else []
        num_fellowships = min(len(fellowships), int(form.fellowship.max_entries))
        form.num_fellowships.data = str(num_fellowships)
        for i in range(num_fellowships):
            fellowship_entry = form.fellowship.append_entry()
            fellowship_entry.data = fellowships[i]
            date_entry = form.fellowship_grad_month_year.append_entry()
            date_entry.data = fellowship_dates[i] if i < len(fellowship_dates) else ''
        while len(form.fellowship.entries) < int(form.fellowship.max_entries):
            form.fellowship.append_entry()
            form.fellowship_grad_month_year.append_entry()

        # Malpractice cases (with limit)
        form.malpractice_cases.entries.clear()
        malpractice_cases = json.loads(doctor.malpractice_cases or '[]')
        num_cases = min(len(malpractice_cases), int(form.malpractice_cases.max_entries))
        form.num_malpractice_cases.data = str(num_cases)
        for case in malpractice_cases[:num_cases]:
            entry = form.malpractice_cases.append_entry()
            entry.incident_year.data = case.get('incident_year', '')
            entry.outcome.data = case.get('outcome', '')
            entry.payout_amount.data = case.get('payout_amount', 0.0)
        while len(form.malpractice_cases.entries) < int(form.malpractice_cases.max_entries):
            form.malpractice_cases.append_entry()

        # NP/PA fields
        form.bachelors.data = doctor.bachelors
        form.bachelors_grad_month_year.data = doctor.bachelors_grad_month_year
        form.msn.data = doctor.msn
        form.msn_grad_month_year.data = doctor.msn_grad_month_year
        form.dnp.data = doctor.dnp
        form.dnp_grad_month_year.data = doctor.dnp_grad_month_year
        form.additional_training.data = doctor.additional_training
        form.sponsorship_needed.data = doctor.sponsorship_needed

        # Certification and additional fields
        form.certification.data = doctor.certification
        form.certification_specialty_area.data = doctor.certification_specialty_area
        form.clinically_active.data = doctor.clinically_active
        form.last_clinically_active.data = doctor.last_clinically_active
        form.emr.data = doctor.emr
        form.languages.data = doctor.languages
        form.states_licensed.data = doctor.states_licensed.split(',') if doctor.states_licensed else []
        form.states_willing_to_work.data = doctor.states_willing_to_work.split(',') if doctor.states_willing_to_work else []
        form.salary_expectations.data = doctor.salary_expectations
            
    return render_template('edit_doctor.html', form=form, doctor=doctor, zip=zip)



with app.app_context():
    db.create_all()  # <-- Create tables first!

    # Check if admin already exists to avoid duplicates
    existing_admin = User.query.filter_by(username='adminchan').first()

    if not existing_admin:
        admin = User(
            username='adminchan',
            email='admin@example.com',
            password_hash=generate_password_hash('icecream2'),
            role='admin'
        )
        db.session.add(admin)
        db.session.commit()
        print("Admin user 'adminchan' created successfully!")
    else:
        print("Admin user already exists.")
# App initialization
threading.Thread(target=lambda: (time.sleep(1), webbrowser.open('http://localhost:5000'))).start()
@app.route('/')
def landing_page():
    return render_template('landing_page.html', current_year=datetime.now().year)

@app.route('/public_register_client', methods=['POST'])
def public_register_client():
    data = request.form

    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    confirm_password = data.get('confirm_password')

    if password != confirm_password:
        flash('Passwords do not match.', 'danger')
        return redirect(url_for('create_account'))

    if User.query.filter((User.username == username) | (User.email == email)).first():
        flash('Username or email already exists.', 'danger')
        return redirect(url_for('create_account'))

    user = User(username=username, email=email, role='client')
    user.set_password(password)
    db.session.add(user)
    db.session.commit()

    login_user(user)
    flash('Client account created and logged in!', 'success')
    return redirect(url_for('client_dashboard'))

@app.route('/client/analytics')
@login_required
def client_analytics():
    if current_user.role != 'client':
        flash('Unauthorized access!', 'danger')
        return redirect(url_for('home'))

    jobs = Job.query.filter_by(poster_id=current_user.id).order_by(Job.id.desc()).all()

    job_data = []
    for job in jobs:
        messages = Message.query.filter_by(job_id=job.id, message_type='interest').all()

        # Count interest per day
        interest_by_day = defaultdict(int)
        interested_doctors = []
        for msg in messages:
            date_str = msg.timestamp.strftime("%Y-%m-%d")
            interest_by_day[date_str] += 1

            if msg.doctor:
                interested_doctors.append({
                    'id': msg.doctor.id,
                    'name': f"{msg.doctor.first_name} {msg.doctor.last_name}",
                    'email': msg.doctor.email
                })

        job_data.append({
            'id': job.id,
            'title': job.title,
            'location': job.location,
            'salary': job.salary,
            'description': job.description,
            'interest_count': len(messages),
            'interested_doctors': interested_doctors,
            'interest_by_day': interest_by_day
        })

    return render_template("client_analytics.html", job_data=job_data)

@app.route('/public_register_doctor', methods=['POST'])
def public_register_doctor():
    data = request.form

    # Extract fields
    username = data.get('username')
    email = data.get('email')
    password = data.get('password')
    confirm_password = data.get('confirm_password')
    first_name = data.get('first_name')
    last_name = data.get('last_name')
    specialty = data.get('specialty')

    # Validation
    if password != confirm_password:
        flash('Passwords do not match.', 'danger')
        return redirect(url_for('create_account'))

    if User.query.filter((User.username == username) | (User.email == email)).first():
        flash('Username or email already exists.', 'danger')
        return redirect(url_for('create_account'))

    # Create user
    user = User(username=username, email=email, role='doctor')
    user.set_password(password)
    db.session.add(user)
    db.session.commit()

    # Create doctor profile
    doctor = Doctor(
        first_name=first_name,
        last_name=last_name,
        email=email,
        specialty=specialty,
        position='MD',  # or allow selection if you want
        user_id=user.id,
        joined=datetime.utcnow()
    )
    db.session.add(doctor)
    db.session.commit()

    login_user(user)
    flash('Account created and logged in!', 'success')
    return redirect(url_for('doctor_dashboard'))
@app.route('/home')
def home():
    return redirect(url_for('dashboard'))

@app.route('/reset_db')
def reset_db():
    with app.app_context():
        db.drop_all()
        db.create_all()

        # Recreate your default admin user
        admin = User(
            username='adminchan',
            email='admin@example.com',
            password_hash=generate_password_hash('icecream2'),
            role='admin'
        )
        db.session.add(admin)
        db.session.commit()

    return "✅ Database has been reset and admin user created!"

with app.app_context():
    db.create_all()

    doctors = Doctor.query.all()
    print("Existing Doctors and Emails:")
    for doc in doctors:
        print(doc.first_name, doc.last_name, doc.email)




# Run the app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

















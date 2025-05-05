import streamlit as st
import torch
import random
import numpy as np
from transformers import pipeline
from sentence_transformers import SentenceTransformer, util

# Page configuration
st.set_page_config(
    page_title="Technical Interview Simulator",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
ROLE_SPECIFIC_TOPICS = {
    "data scientist": [
        "machine learning algorithms", "feature engineering", "data preprocessing",
        "statistical analysis", "hypothesis testing", "model evaluation",
        "deep learning", "natural language processing", "time series analysis",
        "big data technologies", "data visualization", "A/B testing"
    ],
    "software engineer": [
        "data structures", "algorithms", "system design", "object-oriented programming",
        "database design", "API development", "testing strategies", "version control",
        "concurrency", "distributed systems", "microservices", "cloud architecture"
    ],
    "frontend developer": [
        "HTML/CSS", "JavaScript frameworks", "responsive design", "web accessibility",
        "state management", "web performance", "browser compatibility", "API integration",
        "component architecture", "testing frameworks", "build tools", "CSS preprocessors"
    ],
    "backend developer": [
        "server architecture", "RESTful APIs", "database optimization", "authentication",
        "caching strategies", "message queues", "container orchestration", "microservices",
        "security best practices", "logging and monitoring", "ORM frameworks", "scaling strategies"
    ],
    "devops engineer": [
        "CI/CD pipelines", "infrastructure as code", "container orchestration", "monitoring and logging",
        "cloud platforms", "networking", "security practices", "resource optimization",
        "disaster recovery", "automation", "configuration management", "serverless architecture"
    ],
    "machine learning engineer": [
        "model deployment", "MLOps", "feature stores", "model monitoring",
        "data pipelines", "model optimization", "distributed training", "ensemble methods",
        "hyperparameter tuning", "experiment tracking", "model versioning", "edge deployment"
    ],
    "ui/ux designer": [
        "user research", "wireframing", "prototyping", "usability testing",
        "information architecture", "interaction design", "visual design", "design systems",
        "accessibility", "responsive design", "user flows", "A/B testing"
    ]
}

LEVEL_COMPLEXITY = {
    "junior": {
        "depth": "fundamental understanding",
        "complexity": "basic",
        "autonomy": "with guidance",
        "topics": "core concepts"
    },
    "mid": {
        "depth": "practical application",
        "complexity": "moderate",
        "autonomy": "independently",
        "topics": "standard patterns and practices"
    },
    "senior": {
        "depth": "architectural decisions",
        "complexity": "advanced",
        "autonomy": "lead and mentor others",
        "topics": "edge cases, optimizations, and tradeoffs"
    }
}

@st.cache_resource
def load_embedding_model():
    """Load and cache the embedding model"""
    try:
        return SentenceTransformer('all-MiniLM-L6-v2')
    except Exception as e:
        st.error(f"Error loading embedding model: {e}")
        return None

# Initialize session state variables
def init_session_state():
    if 'current_question' not in st.session_state:
        st.session_state.current_question = ""
    if 'ideal_answer' not in st.session_state:
        st.session_state.ideal_answer = ""
    if 'question_number' not in st.session_state:
        st.session_state.question_number = 1
    if 'total_questions' not in st.session_state:
        st.session_state.total_questions = 3
    if 'asked_questions' not in st.session_state:
        st.session_state.asked_questions = []
    if 'scores' not in st.session_state:
        st.session_state.scores = []
    if 'interview_complete' not in st.session_state:
        st.session_state.interview_complete = False
    if 'show_evaluation' not in st.session_state:
        st.session_state.show_evaluation = False
    if 'current_evaluation' not in st.session_state:
        st.session_state.current_evaluation = {}
    if 'job_role' not in st.session_state:
        st.session_state.job_role = ""
    if 'level' not in st.session_state:
        st.session_state.level = ""

def get_role_topics(job_role):
    """Get relevant topics for a specific job role"""
    normalized_role = job_role.lower()
    
    if normalized_role in ROLE_SPECIFIC_TOPICS:
        return ROLE_SPECIFIC_TOPICS[normalized_role]
    
    for role, topics in ROLE_SPECIFIC_TOPICS.items():
        if role in normalized_role or any(word in normalized_role for word in role.split()):
            return topics
    
    return ROLE_SPECIFIC_TOPICS["software engineer"]

def generate_question(job_role, level):
    """Generate a question based on role and level"""
    topics = get_role_topics(job_role)
    selected_topic = random.choice(topics)
    
    # Templates for different levels
    templates = {
        "junior": [
            f"What are the basic principles of {selected_topic}?",
            f"How would you explain {selected_topic} to a non-technical person?",
            f"What common challenges might you face when working with {selected_topic}?",
            f"What are the fundamental concepts of {selected_topic} that every {job_role} should know?"
        ],
        "mid": [
            f"How would you implement {selected_topic} in a real-world project?",
            f"Compare and contrast different approaches to {selected_topic}.",
            f"How do you optimize {selected_topic} for better performance?",
            f"What best practices do you follow when working with {selected_topic}?"
        ],
        "senior": [
            f"How would you architect a system that effectively utilizes {selected_topic} at scale?",
            f"What are the technical tradeoffs to consider when implementing {selected_topic} in an enterprise environment?",
            f"How would you lead a team in implementing {selected_topic} for a critical application?",
            f"Describe how you would solve complex edge cases when dealing with {selected_topic}."
        ]
    }
    
    return random.choice(templates.get(level.lower(), templates["mid"]))

def generate_ideal_answer(question, job_role, level):
    """Generate a model answer based on the question, role and level"""
    # This is a simplified version without the ML model
    topics = get_role_topics(job_role)
    complexity = LEVEL_COMPLEXITY.get(level.lower(), LEVEL_COMPLEXITY["mid"])
    
    # Template answer with role-specific content
    answer = f"""As a {level} {job_role}, here's my approach to this question:

1. Understanding the Core Concepts:
   {random.choice(topics)} is fundamental to addressing this question. When working with {question.split()[4:8]}, it's important to understand {complexity['depth']}.

2. Practical Implementation:
   In a real-world scenario, I would implement this by first analyzing requirements, then designing a solution that considers {random.choice(topics)} and {random.choice(topics)}. 
   
   For example, if building a system for {random.choice(topics)}, I would:
   - Start with clear requirements gathering
   - Design the architecture with {complexity['topics']} in mind
   - Implement the solution using appropriate tools and frameworks
   - Test thoroughly to ensure reliability

3. Optimization and Tradeoffs:
   When optimizing this solution, key considerations include performance, scalability, and maintainability. 
   
   The main tradeoffs to consider are:
   - Speed vs. accuracy
   - Simplicity vs. flexibility
   - Cost vs. performance
   
4. Best Practices:
   Following industry standards, I would ensure proper documentation, testing, and monitoring are in place. 
   For {complexity['complexity']} implementations, it's crucial to follow established patterns while being open to innovation where appropriate.

In summary, success with {question.split()[4:6]} requires strong technical knowledge, practical experience, and the ability to navigate tradeoffs {complexity['autonomy']}.
"""
    return answer

def extract_key_concepts(text):
    """Extract key technical concepts from text"""
    tech_terms = [
        "algorithm", "optimization", "architecture", "database", "framework",
        "pattern", "design", "performance", "scalability", "security",
        "testing", "deployment", "api", "interface", "component",
        "service", "microservice", "container", "cloud", "serverless"
    ]
    
    words = text.lower().split()
    extracted = [term for term in tech_terms if term in text.lower()]
    
    # Add bigrams
    for i in range(len(words)-1):
        bigram = words[i] + " " + words[i+1]
        if any(term in bigram for term in tech_terms):
            extracted.append(bigram)
    
    return list(set(extracted))

def calculate_similarity(ans1, ans2, embedding_model):
    """Calculate semantic similarity between answers"""
    if not ans1.strip() or not ans2.strip():
        return 0.0
    
    try:
        if embedding_model:
            emb1 = embedding_model.encode(ans1, convert_to_tensor=True)
            emb2 = embedding_model.encode(ans2, convert_to_tensor=True)
            
            similarity = util.pytorch_cos_sim(emb1, emb2)
            
            # Convert to Python float if it's a tensor
            if hasattr(similarity, 'item'):
                similarity = similarity.item()
            
            return min(1.0, max(0.0, similarity))
        else:
            # Fallback if model not loaded
            return 0.5
    except Exception as e:
        st.warning(f"Similarity calculation error: {e}")
        return 0.5

def format_feedback_manually(user_answer, ideal_answer):
    """Generate feedback based on comparing answers"""
    user_concepts = extract_key_concepts(user_answer)
    ideal_concepts = extract_key_concepts(ideal_answer)
    
    missing = list(set(ideal_concepts) - set(user_concepts))[:3]
    present = list(set(user_concepts) & set(ideal_concepts))[:3]
    
    feedback = [
        f"Strength: You covered {', '.join(present)} well" if present else "Strength: Clear communication of concepts",
        f"Improvement: Consider adding discussion of {', '.join(missing)}" if missing else "Improvement: Add more technical depth to your response",
        "Suggestion: Structure your answer with clear examples and practical applications"
    ]
    return "\n".join(feedback)

def evaluate_answer(user_answer, ideal_answer, job_role, level, question, embedding_model):
    """Evaluate the user's answer"""
    try:
        similarity = calculate_similarity(user_answer, ideal_answer, embedding_model)
        ideal_concepts = extract_key_concepts(ideal_answer)
        user_concepts = extract_key_concepts(user_answer)
        
        concept_coverage = len(set(user_concepts) & set(ideal_concepts)) / max(len(ideal_concepts), 1)
        completeness = min(1.0, len(user_answer.split()) / 200)
        
        # Calculate score
        score = 0.5 * similarity + 0.3 * concept_coverage + 0.2 * completeness
        score = min(1.0, max(0.1, score))
        
        # Generate feedback
        strengths = []
        if similarity >= 0.7:
            strengths.append("Strong alignment with key concepts")
        if concept_coverage >= 0.7:
            strengths.append("Good coverage of technical topics")
        if completeness >= 0.8:
            strengths.append("Comprehensive response")
        if not strengths:
            strengths.append("Clear communication")
            
        improvements = []
        if similarity < 0.7:
            improvements.append("Better align with industry standards")
        if concept_coverage < 0.7:
            missing = list(set(ideal_concepts) - set(user_concepts))[:3]
            if missing:
                improvements.append(f"Discuss: {', '.join(missing)}")
        if "tradeoff" not in user_answer.lower():
            improvements.append("Discuss tradeoffs between approaches")
        if not improvements:
            improvements.append("Add more technical depth")
            
        if score >= 0.85:
            assessment = "Excellent technical response!"
        elif score >= 0.75:
            assessment = "Strong answer with minor improvements needed"
        elif score >= 0.65:
            assessment = "Good answer, some areas to develop"
        elif score >= 0.5:
            assessment = "Basic understanding shown"
        else:
            assessment = "Needs significant improvement"
            
        feedback = format_feedback_manually(user_answer, ideal_answer)
        
        return {
            "score": score,
            "assessment": assessment,
            "strengths": strengths,
            "improvements": improvements,
            "feedback": feedback,
            "model_answer": ideal_answer
        }
        
    except Exception as e:
        st.error(f"Evaluation error: {e}")
        return {
            "score": 0.5,
            "assessment": "Evaluation unavailable",
            "strengths": [],
            "improvements": ["System error occurred"],
            "feedback": "Could not generate feedback",
            "model_answer": ideal_answer
        }

def generate_new_question():
    """Generate a new question and reset the evaluation state"""
    if st.session_state.question_number <= st.session_state.total_questions:
        question = generate_question(st.session_state.job_role, st.session_state.level)
        
        # Make sure we don't repeat questions
        attempts = 0
        while question in st.session_state.asked_questions and attempts < 10:
            question = generate_question(st.session_state.job_role, st.session_state.level)
            attempts += 1
            
        st.session_state.current_question = question
        st.session_state.asked_questions.append(question)
        st.session_state.ideal_answer = generate_ideal_answer(
            question, st.session_state.job_role, st.session_state.level
        )
        st.session_state.show_evaluation = False
    else:
        st.session_state.interview_complete = True

def evaluate_current_answer():
    """Evaluate the current answer and show results"""
    user_answer = st.session_state.user_answer
    embedding_model = load_embedding_model()
    
    evaluation = evaluate_answer(
        user_answer,
        st.session_state.ideal_answer,
        st.session_state.job_role,
        st.session_state.level,
        st.session_state.current_question,
        embedding_model
    )
    
    st.session_state.current_evaluation = evaluation
    st.session_state.scores.append(evaluation["score"])
    st.session_state.show_evaluation = True
    st.session_state.question_number += 1

def skip_question():
    """Skip the current question"""
    st.session_state.question_number += 1
    generate_new_question()

def restart_interview():
    """Reset all session state to start a new interview"""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    init_session_state()

def main():
    # Initialize session state
    init_session_state()
    
    # App title and description
    st.title("🚀 Technical Interview Simulator")
    
    # Setup page
    if not st.session_state.job_role:
        st.subheader("Setup Your Interview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            role_options = list(ROLE_SPECIFIC_TOPICS.keys())
            selected_role = st.selectbox(
                "Select your job role:", 
                options=role_options,
                index=role_options.index("software engineer") if "software engineer" in role_options else 0
            )
            st.session_state.job_role = selected_role
            
        with col2:
            level_options = list(LEVEL_COMPLEXITY.keys())
            selected_level = st.selectbox(
                "Select your experience level:",
                options=level_options,
                index=level_options.index("mid") if "mid" in level_options else 0
            )
            st.session_state.level = selected_level
        
        st.session_state.total_questions = st.slider(
            "Number of questions:", 
            min_value=1, 
            max_value=10, 
            value=3
        )
        
        if st.button("Start Interview", type="primary", use_container_width=True):
            generate_new_question()
    
    # Interview in progress
    elif not st.session_state.interview_complete:
        # Display progress
        progress_text = f"Question {st.session_state.question_number}/{st.session_state.total_questions} " + \
                        f"| {st.session_state.level.capitalize()} {st.session_state.job_role}"
        st.progress(st.session_state.question_number / st.session_state.total_questions)
        st.write(progress_text)
        
        # Display current question
        st.subheader("Question:")
        st.write(st.session_state.current_question)
        
        # Answer area
        if not st.session_state.show_evaluation:
            st.text_area(
                "Your Answer:", 
                key="user_answer", 
                height=200,
                placeholder="Type your answer here..."
            )
            
            col1, col2 = st.columns(2)
            with col1:
                submit = st.button("Submit Answer", type="primary", use_container_width=True)
                if submit:
                    evaluate_current_answer()
            with col2:
                skip = st.button("Skip Question", use_container_width=True)
                if skip:
                    skip_question()
        
        # Show evaluation
        if st.session_state.show_evaluation:
            evaluation = st.session_state.current_evaluation
            
            st.subheader("Evaluation Results")
            
            # Score display
            score_col1, score_col2 = st.columns([1, 3])
            with score_col1:
                st.metric("Score", f"{evaluation['score']:.2f}/1.00")
            with score_col2:
                st.write(f"**Assessment:** {evaluation['assessment']}")
            
            # Feedback tabs
            tab1, tab2, tab3 = st.tabs(["Feedback", "Model Answer", "Your Answer"])
            
            with tab1:
                st.subheader("Strengths")
                for s in evaluation['strengths']:
                    st.write(f"✅ {s}")
                    
                st.subheader("Areas for Improvement")
                for i in evaluation['improvements']:
                    st.write(f"🔍 {i}")
                    
                st.subheader("Technical Feedback")
                st.write(evaluation['feedback'])
            
            with tab2:
                st.write(evaluation['model_answer'])
            
            with tab3:
                st.write(st.session_state.user_answer)
            
            # Next question button
            if st.session_state.question_number <= st.session_state.total_questions:
                if st.button("Next Question", type="primary", use_container_width=True):
                    generate_new_question()
            else:
                if st.button("See Final Results", type="primary", use_container_width=True):
                    st.session_state.interview_complete = True
    
    # Interview complete
    else:
        st.subheader("🎉 Interview Complete!")
        
        # Calculate overall score
        avg_score = sum(st.session_state.scores) / len(st.session_state.scores) if st.session_state.scores else 0
        
        # Display final score
        st.metric("Overall Score", f"{avg_score:.2f}/1.00")
        
        # Assessment based on score
        if avg_score >= 0.8:
            assessment = "Excellent performance! You're well prepared for interviews."
        elif avg_score >= 0.7:
            assessment = "Strong performance with some areas to polish."
        elif avg_score >= 0.6:
            assessment = "Good fundamentals, keep practicing."
        else:
            assessment = "Keep studying and try again."
        
        st.write(f"**Assessment:** {assessment}")
        
        # Topics to review
        st.subheader("Recommended Topics to Review:")
        topics = get_role_topics(st.session_state.job_role)
        for topic in random.sample(topics, min(3, len(topics))):
            st.write(f"- {topic}")
        
        # Restart button
        if st.button("Start New Interview", type="primary", use_container_width=True):
            restart_interview()
    
    # Footer
    st.sidebar.header("About")
    st.sidebar.info(
        "This Technical Interview Simulator helps you practice for tech interviews "
        "by generating role-specific questions and providing feedback on your answers."
    )
    
    st.sidebar.header("Available Roles")
    role_text = ", ".join(ROLE_SPECIFIC_TOPICS.keys())
    st.sidebar.write(role_text)
    
    if st.session_state.job_role:
        st.sidebar.header("Current Topics")
        topics = get_role_topics(st.session_state.job_role)
        for topic in random.sample(topics, min(5, len(topics))):
            st.sidebar.write(f"- {topic}")

if __name__ == "__main__":
    main()
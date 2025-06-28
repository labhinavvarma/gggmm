def display_progressive_workflow_animation():
    """Display progressive workflow animation using Streamlit native components"""
    try:
        # Calculate progress percentage
        completed_steps = sum(1 for step in st.session_state.workflow_steps.values() if step['status'] == 'completed')
        progress_percentage = (completed_steps / len(st.session_state.workflow_steps)) * 100
        
        # Use Streamlit native components instead of HTML
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    padding: 2rem; border-radius: 12px; margin: 2rem 0; color: white;
                    box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);">
        """, unsafe_allow_html=True)
        
        # Header
        st.markdown("<h3 style='text-align: center; color: white; margin-bottom: 1rem;'>🔬 Deep Research Analysis</h3>", unsafe_allow_html=True)
        
        # Step counter
        st.markdown(f"""
        <div style="text-align: center; background: rgba(255, 255, 255, 0.15); 
                    border-radius: 20px; padding: 0.5rem 1rem; display: inline-block; 
                    font-size: 0.9rem; font-weight: 600; margin-bottom: 1rem;">
            Step {st.session_state.current_step} of 8
        </div>
        """, unsafe_allow_html=True)
        
        # Progress bar using Streamlit's native progress bar
        st.progress(progress_percentage / 100)
        
        # Display steps using Streamlit columns and native components
        steps_revealed = st.session_state.steps_revealed
        
        for step_num in range(1, min(steps_revealed + 2, 9)):
            if step_num in st.session_state.workflow_steps:
                step_info = st.session_state.workflow_steps[step_num]
                status = step_info['status']
                name = step_info['name']
                description = step_info.get('description', '')
                
                # Create columns for icon and text
                col1, col2 = st.columns([1, 8])
                
                with col1:
                    if status == 'pending':
                        st.markdown(f"""
                        <div style="width: 40px; height: 40px; border-radius: 50%; 
                                    background: rgba(255, 255, 255, 0.1); color: rgba(255, 255, 255, 0.5);
                                    border: 2px solid rgba(255, 255, 255, 0.2); display: flex; 
                                    align-items: center; justify-content: center; font-weight: bold;">
                            {step_num}
                        </div>
                        """, unsafe_allow_html=True)
                    elif status == 'running':
                        st.markdown("""
                        <div style="width: 40px; height: 40px; border-radius: 50%; 
                                    background: #ffc107; color: #000; border: 2px solid #ffca2c;
                                    display: flex; align-items: center; justify-content: center; 
                                    font-weight: bold; animation: pulse 1s infinite;">
                            ●
                        </div>
                        """, unsafe_allow_html=True)
                    elif status == 'completed':
                        st.markdown("""
                        <div style="width: 40px; height: 40px; border-radius: 50%; 
                                    background: #28a745; color: white; border: 2px solid #34ce57;
                                    display: flex; align-items: center; justify-content: center; 
                                    font-weight: bold;">
                            ✓
                        </div>
                        """, unsafe_allow_html=True)
                    elif status == 'error':
                        st.markdown("""
                        <div style="width: 40px; height: 40px; border-radius: 50%; 
                                    background: #dc3545; color: white; border: 2px solid #e4606d;
                                    display: flex; align-items: center; justify-content: center; 
                                    font-weight: bold;">
                            ✗
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    # Step name and description
                    if status == 'running':
                        st.markdown(f"**{name}** ⚡", help=description)
                        st.caption(description)
                    elif status == 'completed':
                        st.markdown(f"~~{name}~~ ✅", help=description)
                        st.caption(f"✅ {description}")
                    elif status == 'error':
                        st.markdown(f"**{name}** ❌", help=description)
                        st.caption(f"❌ {description}")
                    else:
                        st.markdown(f"{name}", help=description)
                        st.caption(description)
                
                # Add some spacing
                st.write("")
        
        # Show preview of next step
        if steps_revealed < len(st.session_state.workflow_steps) and st.session_state.current_step > 0:
            next_step_num = steps_revealed + 1
            if next_step_num <= len(st.session_state.workflow_steps):
                next_step_name = st.session_state.workflow_steps[next_step_num]['name']
                st.info(f"🔄 Next: {next_step_name}...")
        
        # Status message
        st.markdown("""
        <div style="text-align: center; margin-top: 1.5rem; font-style: italic; 
                    opacity: 0.9; font-size: 0.9rem; color: white;">
            Comprehensive analysis in progress...
        </div>
        """, unsafe_allow_html=True)
        
        # Close the container
        st.markdown("</div>", unsafe_allow_html=True)
        
    except Exception as e:
        logger.warning(f"Error creating progressive animation: {e}")
        # Fallback simple display
        st.info("🔬 Deep Research Analysis in Progress...")
        st.progress(st.session_state.current_step / 8)
        if st.session_state.current_step > 0:
            current_step_name = st.session_state.workflow_steps[st.session_state.current_step]['name']
            st.write(f"Current step: {current_step_name}")

# Alternative simpler version if the above still has issues
def display_simple_workflow_animation():
    """Simple workflow animation using only Streamlit native components"""
    
    # Header
    st.markdown("### 🔬 Deep Research Analysis")
    
    # Progress
    completed_steps = sum(1 for step in st.session_state.workflow_steps.values() if step['status'] == 'completed')
    progress_percentage = completed_steps / len(st.session_state.workflow_steps)
    
    st.progress(progress_percentage, text=f"Step {st.session_state.current_step} of 8")
    
    # Current step info
    if st.session_state.current_step > 0:
        current_step = st.session_state.workflow_steps[st.session_state.current_step]
        
        if current_step['status'] == 'running':
            st.info(f"🔄 **{current_step['name']}** - {current_step['description']}")
        elif current_step['status'] == 'completed':
            st.success(f"✅ **{current_step['name']}** - Completed")
        elif current_step['status'] == 'error':
            st.error(f"❌ **{current_step['name']}** - Error")
    
    # Show completed steps
    st.markdown("**Completed Steps:**")
    completed_step_names = []
    for step_num, step_info in st.session_state.workflow_steps.items():
        if step_info['status'] == 'completed':
            completed_step_names.append(f"✅ {step_info['name']}")
    
    if completed_step_names:
        for step_name in completed_step_names:
            st.write(step_name)
    else:
        st.write("No steps completed yet...")
    
    # Show what's coming next
    if st.session_state.current_step < len(st.session_state.workflow_steps):
        next_step_num = st.session_state.current_step + 1
        if next_step_num in st.session_state.workflow_steps:
            next_step = st.session_state.workflow_steps[next_step_num]
            st.write(f"🔄 **Next:** {next_step['name']}")

# Updated animation display call - USE ONE OF THESE OPTIONS:

# OPTION 1: Enhanced version (try this first)
def show_animation_enhanced():
    """Show the enhanced animation"""
    if st.session_state.analysis_running and st.session_state.show_animation:
        with animation_container.container():
            display_progressive_workflow_animation()

# OPTION 2: Simple fallback version (use if Option 1 has issues)
def show_animation_simple():
    """Show the simple animation"""
    if st.session_state.analysis_running and st.session_state.show_animation:
        with animation_container.container():
            display_simple_workflow_animation()

# ================================================================
# INTEGRATION INSTRUCTIONS:
# ================================================================

# In your main code, replace this line:
# st.markdown(display_progressive_workflow_animation(), unsafe_allow_html=True)

# WITH ONE OF THESE OPTIONS:

# OPTION 1: Try the enhanced version first
if st.session_state.analysis_running and st.session_state.show_animation:
    with animation_container.container():
        display_progressive_workflow_animation()

# OPTION 2: If Option 1 shows HTML code, use this simpler version
# if st.session_state.analysis_running and st.session_state.show_animation:
#     with animation_container.container():
#         display_simple_workflow_animation()

# ================================================================
# COMPLETE UPDATED ANIMATION SECTION FOR YOUR CODE:
# ================================================================

"""
Replace your animation display section with this:

# Analysis Status Display with Progressive Animation
animation_container = st.empty()

# Show animation during analysis
if st.session_state.analysis_running and st.session_state.show_animation:
    with animation_container.container():
        display_progressive_workflow_animation()  # Use the enhanced version
        
# Clear animation when done
elif not st.session_state.analysis_running and st.session_state.animation_complete:
    animation_container.empty()
"""

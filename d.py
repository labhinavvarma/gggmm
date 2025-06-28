
# Replace the existing CSS section with this enhanced version
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    color: #2c3e50;
    text-align: center;
    margin-bottom: 2rem;
    font-weight: 600;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.section-box {
    background: white;
    padding: 1.5rem;
    border-radius: 12px;
    border: 1px solid #e9ecef;
    margin: 1rem 0;
    box-shadow: 0 4px 15px rgba(0,0,0,0.1);
}

.section-title {
    font-size: 1.3rem;
    color: #2c3e50;
    font-weight: 600;
    margin-bottom: 1rem;
    border-bottom: 2px solid #3498db;
    padding-bottom: 0.5rem;
}

/* Green Run Analysis Button */
.stButton button[kind="primary"] {
    background: linear-gradient(135deg, #28a745 0%, #20c997 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 0.75rem 2rem !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 15px rgba(40, 167, 69, 0.3) !important;
    transition: all 0.3s ease !important;
}

.stButton button[kind="primary"]:hover {
    background: linear-gradient(135deg, #218838 0%, #1abc9c 100%) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(40, 167, 69, 0.4) !important;
}

/* Enhanced Deep Research Animation Styles */
.enhanced-research-container {
    background: linear-gradient(135deg, #0f1419 0%, #1a2332 50%, #0d1117 100%);
    padding: 2.5rem;
    border-radius: 16px;
    margin: 2rem 0;
    color: white;
    box-shadow: 0 12px 40px rgba(15, 20, 25, 0.8);
    border: 1px solid rgba(255, 255, 255, 0.1);
    animation: containerGlow 3s ease-in-out infinite;
    position: relative;
    overflow: hidden;
}

.enhanced-research-container::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.1), transparent);
    animation: scan 2s infinite;
}

.research-header {
    text-align: center;
    margin-bottom: 2rem;
    border-bottom: 1px solid rgba(255, 255, 255, 0.2);
    padding-bottom: 1.5rem;
}

.research-header h2 {
    margin: 0 0 1rem 0;
    font-size: 1.8rem;
    font-weight: 700;
    background: linear-gradient(135deg, #00d4ff 0%, #90e0ff 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.research-status {
    margin-top: 0.5rem;
}

.status-badge {
    background: linear-gradient(135deg, #00d4aa 0%, #00d4ff 100%);
    color: #000;
    padding: 0.5rem 1.5rem;
    border-radius: 20px;
    font-size: 0.85rem;
    font-weight: 600;
    display: inline-block;
    animation: badgePulse 2s infinite;
}

.progress-section {
    margin: 2rem 0;
    background: rgba(255, 255, 255, 0.05);
    padding: 1.5rem;
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.progress-info {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
    font-size: 0.95rem;
}

.progress-text {
    font-weight: 600;
    color: #00d4ff;
}

.time-estimate {
    color: #90e0ff;
    font-weight: 500;
}

.progress-bar {
    width: 100%;
    height: 8px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 4px;
    overflow: hidden;
    margin-bottom: 0.5rem;
}

.progress-fill {
    height: 100%;
    background: linear-gradient(90deg, #00d4aa 0%, #00d4ff 100%);
    border-radius: 4px;
    transition: width 0.8s ease;
    animation: progressShimmer 2s infinite;
}

.progress-percentage {
    text-align: center;
    font-size: 0.9rem;
    color: #90e0ff;
    font-weight: 600;
}

.workflow-steps {
    margin: 2rem 0;
}

.research-step {
    display: flex;
    align-items: flex-start;
    padding: 1.25rem 0;
    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    transition: all 0.5s ease;
}

.research-step:last-child {
    border-bottom: none;
}

.research-step-icon {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    margin-right: 1.5rem;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 16px;
    font-weight: bold;
    transition: all 0.4s ease;
    border: 2px solid transparent;
    flex-shrink: 0;
}

.research-step-content {
    flex: 1;
    min-width: 0;
}

.research-step-text {
    font-weight: 600;
    font-size: 1.1rem;
    margin-bottom: 0.3rem;
    transition: all 0.3s ease;
}

.research-step-description {
    font-size: 0.9rem;
    opacity: 0.8;
    font-style: italic;
    transition: all 0.3s ease;
}

.step-pending .research-step-icon {
    background: rgba(255, 255, 255, 0.1);
    color: rgba(255, 255, 255, 0.5);
    border-color: rgba(255, 255, 255, 0.2);
}

.step-running .research-step-icon {
    background: linear-gradient(135deg, #ffc107 0%, #ff8c00 100%);
    color: #000;
    border-color: #ffd700;
    animation: iconPulse 1s infinite;
    box-shadow: 0 0 20px rgba(255, 193, 7, 0.6);
}

.step-running .research-step-text {
    color: #ffd700;
    font-weight: 700;
}

.step-running .research-step-description {
    color: #ffeb3b;
    opacity: 1;
}

.step-completed .research-step-icon {
    background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
    color: white;
    border-color: #34ce57;
    box-shadow: 0 0 15px rgba(40, 167, 69, 0.4);
}

.step-completed .research-step-text {
    opacity: 0.9;
    color: #90e0ff;
}

.step-completed .research-step-description {
    opacity: 0.7;
}

.step-error .research-step-icon {
    background: linear-gradient(135deg, #dc3545 0%, #c82333 100%);
    color: white;
    border-color: #e4606d;
}

.research-footer {
    text-align: center;
    margin-top: 2rem;
    padding-top: 1.5rem;
    border-top: 1px solid rgba(255, 255, 255, 0.2);
    position: relative;
}

.neural-pulse {
    width: 60px;
    height: 60px;
    border: 2px solid #00d4ff;
    border-radius: 50%;
    margin: 0 auto 1rem auto;
    animation: neuralPulse 2s infinite;
    position: relative;
}

.neural-pulse::before {
    content: '';
    position: absolute;
    top: 50%;
    left: 50%;
    transform: translate(-50%, -50%);
    width: 8px;
    height: 8px;
    background: #00d4ff;
    border-radius: 50%;
}

.research-note {
    font-size: 0.95rem;
    color: #90e0ff;
    font-style: italic;
}

/* Minimized Success State */
.minimized-success-container {
    background: linear-gradient(135deg, #28a745 0%, #20c997 100%);
    padding: 1.5rem;
    border-radius: 12px;
    margin: 1rem 0;
    color: white;
    box-shadow: 0 8px 25px rgba(40, 167, 69, 0.4);
    animation: successGlow 2s ease-in-out infinite;
    display: flex;
    align-items: center;
    justify-content: space-between;
    cursor: pointer;
    transition: all 0.3s ease;
}

.minimized-success-container:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 35px rgba(40, 167, 69, 0.6);
}

.success-content {
    display: flex;
    align-items: center;
    flex: 1;
}

.success-icon {
    font-size: 2.5rem;
    margin-right: 1.5rem;
    animation: bounceSuccess 1s ease-out;
}

.success-text h3 {
    margin: 0 0 0.5rem 0;
    font-size: 1.3rem;
    font-weight: 700;
}

.success-text p {
    margin: 0;
    opacity: 0.9;
    font-size: 0.95rem;
}

.success-stats {
    display: flex;
    gap: 1.5rem;
    margin-left: 2rem;
}

.stat-item {
    text-align: center;
    padding: 0.5rem;
    background: rgba(255, 255, 255, 0.2);
    border-radius: 8px;
    min-width: 60px;
}

.stat-number {
    display: block;
    font-size: 1.2rem;
    font-weight: 700;
    margin-bottom: 0.2rem;
}

.stat-label {
    display: block;
    font-size: 0.7rem;
    opacity: 0.9;
}

.expand-button {
    background: rgba(255, 255, 255, 0.2);
    border: 1px solid rgba(255, 255, 255, 0.3);
    padding: 0.75rem 1.5rem;
    border-radius: 8px;
    cursor: pointer;
    transition: all 0.3s ease;
    font-weight: 600;
    white-space: nowrap;
}

.expand-button:hover {
    background: rgba(255, 255, 255, 0.3);
    transform: translateY(-2px);
}

/* Additional utility styles */
.metric-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 1rem;
    margin: 1rem 0;
}

.metric-card {
    background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
    padding: 1rem;
    border-radius: 8px;
    text-align: center;
    border: 1px solid #dee2e6;
}

.json-container {
    background: #f8f9fa;
    padding: 1rem;
    border-radius: 8px;
    border: 1px solid #dee2e6;
    max-height: 400px;
    overflow-y: auto;
    font-family: monospace;
    font-size: 0.85rem;
}

/* Animations */
@keyframes containerGlow {
    0%, 100% { 
        box-shadow: 0 12px 40px rgba(15, 20, 25, 0.8);
    }
    50% { 
        box-shadow: 0 16px 50px rgba(0, 212, 255, 0.3);
    }
}

@keyframes scan {
    0% { left: -100%; }
    100% { left: 100%; }
}

@keyframes badgePulse {
    0%, 100% { 
        transform: scale(1);
        box-shadow: 0 0 10px rgba(0, 212, 170, 0.5);
    }
    50% { 
        transform: scale(1.05);
        box-shadow: 0 0 20px rgba(0, 212, 170, 0.8);
    }
}

@keyframes progressShimmer {
    0% { 
        background: linear-gradient(90deg, #00d4aa 0%, #00d4ff 100%);
    }
    50% { 
        background: linear-gradient(90deg, #00d4ff 0%, #90e0ff 100%);
    }
    100% { 
        background: linear-gradient(90deg, #00d4aa 0%, #00d4ff 100%);
    }
}

@keyframes iconPulse {
    0%, 100% { 
        transform: scale(1);
        box-shadow: 0 0 20px rgba(255, 193, 7, 0.6);
    }
    50% { 
        transform: scale(1.15);
        box-shadow: 0 0 30px rgba(255, 193, 7, 0.9);
    }
}

@keyframes neuralPulse {
    0%, 100% { 
        transform: scale(1);
        opacity: 1;
    }
    50% { 
        transform: scale(1.2);
        opacity: 0.7;
    }
}

@keyframes successGlow {
    0%, 100% { 
        box-shadow: 0 8px 25px rgba(40, 167, 69, 0.4);
    }
    50% { 
        box-shadow: 0 12px 35px rgba(40, 167, 69, 0.6);
    }
}

@keyframes bounceSuccess {
    0% { 
        transform: scale(0);
    }
    50% { 
        transform: scale(1.2);
    }
    100% { 
        transform: scale(1);
    }
}
</style>
""", unsafe_allow_html=True)

# Replace the animation display logic in your main app
# Animation Status Display with Enhanced Animation
animation_container = st.empty()  # Declare at global scope

# Enhanced animation display logic
if st.session_state.analysis_running and st.session_state.show_animation:
    # Show enhanced workflow animation
    with animation_container.container():
        st.markdown(display_enhanced_workflow_animation(), unsafe_allow_html=True)
        
elif st.session_state.analysis_results and st.session_state.animation_completed:
    # Show minimized success state
    with animation_container.container():
        if st.button("", key="success_summary", help="Click to expand details"):
            st.session_state.show_minimized_success = not st.session_state.show_minimized_success
        
        st.markdown(display_minimized_success_animation(), unsafe_allow_html=True)
        
        # Option to show full workflow details
        if st.session_state.show_minimized_success:
            with st.expander("🔍 View Complete Workflow Details", expanded=False):
                st.markdown("### 🔬 Deep Research Analysis Steps Completed:")
                
                for step_num, step_info in st.session_state.workflow_steps.items():
                    status_icon = "✅" if step_info['status'] == 'completed' else "❌"
                    st.markdown(f"{status_icon} **Step {step_num}:** {step_info['name']}")
                    st.markdown(f"   *{step_info['description']}*")
                
                st.markdown(f"**Total Analysis Time:** {st.session_state.total_analysis_time:.1f} seconds")
                st.markdown(f"**Analysis Completed:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

elif st.session_state.analysis_results:
    # Clear animation and show simple results
    animation_container.empty()
    if st.session_state.analysis_results.get("success", False):
        st.success("✅ Deep Research Analysis completed successfully!")
    else:
        st.error("❌ Deep Research Analysis encountered some issues.")

# Enhanced analysis execution with improved workflow steps
if submitted and not st.session_state.analysis_running:
    # Prepare patient data
    patient_data = {
        "first_name": first_name.strip(),
        "last_name": last_name.strip(),
        "ssn": ssn.strip(),
        "date_of_birth": date_of_birth.strftime("%Y-%m-%d"),
        "gender": gender,
        "zip_code": zip_code.strip()
    }
    
    # Validate patient data
    is_valid, validation_errors = validate_patient_data(patient_data)
    
    if not is_valid:
        st.error("❌ Please fix the following errors:")
        for error in validation_errors:
            st.error(f"• {error}")
    else:
        # Initialize Health Agent
        if st.session_state.agent is None:
            try:
                config = st.session_state.config or Config()
                st.session_state.agent = HealthAnalysisAgent(config)
                st.success("✅ Deep Research Health Agent initialized successfully")
                        
            except Exception as e:
                st.error(f"❌ Failed to initialize Deep Research Health Agent: {str(e)}")
                st.stop()
        
        # Start analysis with enhanced animation
        st.session_state.analysis_running = True
        st.session_state.show_animation = True
        st.session_state.animation_completed = False
        st.session_state.show_minimized_success = False
        
        # Reset and initialize workflow animation
        reset_workflow_steps()
        
        # Show starting message
        st.info("🚀 Initiating Deep Research Analysis Protocol...")
        
        with st.spinner("🔬 Deep Research Analysis executing..."):
            try:
                # Execute analysis with step-by-step animation
                for step_num in range(1, 9):
                    update_workflow_step(step_num, 'running')
                    time.sleep(0.3)  # Visual feedback delay
                    
                    # Simulate different step durations
                    if step_num == 1:  # Initialization
                        time.sleep(0.5)
                    elif step_num in [2, 3]:  # Data fetching and deidentification
                        time.sleep(0.8)
                    elif step_num in [4, 5]:  # Extraction and analysis
                        time.sleep(1.0)
                    elif step_num == 6:  # Health trajectory
                        time.sleep(0.7)
                    elif step_num == 7:  # Heart attack prediction
                        time.sleep(0.6)
                    elif step_num == 8:  # Chatbot initialization
                        time.sleep(0.4)
                    
                    update_workflow_step(step_num, 'completed')
                
                # Execute the actual analysis after animation
                results = st.session_state.agent.run_analysis(patient_data)
                
                # Process results
                if results.get("success", False):
                    # Mark analysis as completed and store results
                    st.session_state.analysis_results = results
                    st.session_state.chatbot_context = results.get("chatbot_context", {})
                    st.session_state.animation_completed = True
                    
                    # Show completion message
                    st.balloons()  # Celebration animation
                    st.success("🎉 Deep Research Analysis completed successfully!")
                    
                    # Setup chatbot if ready
                    if results.get("chatbot_ready", False) and st.session_state.chatbot_context:
                        st.success("💬 Medical Assistant is now active with comprehensive data access!")
                        # Force page refresh to show minimized success state
                        time.sleep(1)
                        st.rerun()
                        
                else:
                    # Handle analysis failure
                    st.session_state.analysis_results = results
                    st.warning("⚠️ Analysis completed with some errors.")
                
            except Exception as e:
                # Mark current step as error
                if st.session_state.current_step > 0:
                    update_workflow_step(st.session_state.current_step, 'error')
                
                logger.error(f"Deep research analysis failed: {str(e)}")
                st.error(f"❌ Deep research analysis failed: {str(e)}")
                st.session_state.analysis_results = {
                    "success": False,
                    "error": str(e),
                    "patient_data": patient_data,
                    "errors": [str(e)]
                }
            finally:
                st.session_state.analysis_running = False
                st.session_state.show_animation = False

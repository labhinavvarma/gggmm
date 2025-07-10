Workflow Trigger Sequence
1. Button Click Event
javascript// The green button that starts everything
<button
  onClick={runHealthAnalysis}
  disabled={analysisRunning}
  className="w-full mt-6 px-6 py-3 bg-gradient-to-r from-green-500 to-emerald-600..."
>
  🔬 Run Deep Research Analysis
</button>
2. Analysis Function Execution
javascriptconst runHealthAnalysis = async () => {
  // 1. Validate patient data first
  const { isValid, errors } = validatePatientData(patientData);
  
  if (!isValid) return; // Stop if validation fails
  
  // 2. START WORKFLOW ANIMATION
  setAnalysisRunning(true);     // Shows loading state
  setShowWorkflow(true);        // Displays workflow container
  
  // 3. Begin step-by-step animation
  await runWorkflowSteps();     // THIS IS WHERE LANGRAPH STARTS
}
3. LangGraph Workflow Steps
javascriptconst runWorkflowSteps = async () => {
  for (let i = 0; i < workflowSteps.length; i++) {
    // Set current step to "running"
    steps[i].status = 'running';
    setWorkflowSteps([...steps]);
    
    // Animate for 2.5 seconds
    await new Promise(resolve => setTimeout(resolve, 2500));
    
    // Mark as "completed"
    steps[i].status = 'completed';
    setWorkflowSteps([...steps]);
  }
}

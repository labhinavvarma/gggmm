def display_batch_code_meanings_enhanced(results):
    """Enhanced batch processed code meanings in organized tabular format with proper subdivisions"""
    st.markdown("""
    <div class="batch-meanings-card">
        <h3>🧠 Enhanced Batch Code Meanings Analysis</h3>
        <p><strong>Features:</strong> LLM-powered interpretation of medical and pharmacy codes with detailed tabular display</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Get extraction results
    medical_extraction = safe_get(results, 'structured_extractions', {}).get('medical', {})
    pharmacy_extraction = safe_get(results, 'structured_extractions', {}).get('pharmacy', {})
    
    # Create main tabs for Medical and Pharmacy
    tab1, tab2 = st.tabs(["🏥 Medical Code Meanings", "💊 Pharmacy Code Meanings"])
    
    with tab1:
        st.markdown('<div class="medical-codes-section">', unsafe_allow_html=True)
        st.markdown("### 🏥 Medical Code Meanings Analysis")
        
        medical_meanings = medical_extraction.get("code_meanings", {})
        service_meanings = medical_meanings.get("service_code_meanings", {})
        diagnosis_meanings = medical_meanings.get("diagnosis_code_meanings", {})
        medical_records = medical_extraction.get("hlth_srvc_records", [])
        
        # Medical summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="metric-summary-box">', unsafe_allow_html=True)
            st.metric("Service Codes", len(service_meanings))
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-summary-box">', unsafe_allow_html=True)
            st.metric("ICD-10 Codes", len(diagnosis_meanings))
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-summary-box">', unsafe_allow_html=True)
            st.metric("Medical Records", len(medical_records))
            st.markdown('</div>', unsafe_allow_html=True)
        with col4:
            st.markdown('<div class="metric-summary-box">', unsafe_allow_html=True)
            batch_status = medical_extraction.get("llm_call_status", "unknown")
            st.metric("Batch Status", batch_status)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Create sub-tabs for different medical code types
        med_tab1, med_tab2 = st.tabs(["🩺 ICD-10 Diagnosis Codes", "🏥 Medical Service Codes"])
        
        with med_tab1:
            st.markdown('<div class="code-table-container">', unsafe_allow_html=True)
            st.markdown("#### 🩺 ICD-10 Diagnosis Codes with Dates and Meanings")
            
            if medical_records:
                # Prepare data for enhanced table display - INCLUDE ALL CODES
                diagnosis_data = []
                for record in medical_records:
                    claim_date = record.get("clm_rcvd_dt", "Unknown")
                    service_end_date = record.get("clm_line_srvc_end_dt", "Unknown")
                    record_path = record.get("data_path", "")
                    
                    for diag in record.get("diagnosis_codes", []):
                        code = diag.get("code", "")
                        if code:  # Include ALL codes, not just those with meanings
                            # Use generated meaning if available, otherwise provide fallback
                            code_meaning = diagnosis_meanings.get(code, f"ICD-10 diagnosis code {code}")
                            
                            diagnosis_data.append({
                                "ICD-10 Code": code,
                                "Code Meaning": code_meaning,
                                "Claim Date": claim_date,
                                "Service End Date": service_end_date,
                                "Position": diag.get("position", ""),
                                "Source Field": diag.get("source", ""),
                                "Record Path": record_path,
                                "Meaning Generated": "Yes" if code in diagnosis_meanings else "Fallback"
                            })
                
                if diagnosis_data:
                    # Display all codes count and statistics
                    unique_codes = len(set(item["ICD-10 Code"] for item in diagnosis_data))
                    codes_with_meanings = len([item for item in diagnosis_data if item["Meaning Generated"] == "Yes"])
                    codes_with_fallback = len([item for item in diagnosis_data if item["Meaning Generated"] == "Fallback"])
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📊 Unique ICD-10 Codes", unique_codes)
                    with col2:
                        st.metric("🧠 LLM Generated Meanings", codes_with_meanings)
                    with col3:
                        st.metric("🔄 Fallback Descriptions", codes_with_fallback)
                    
                    # Create DataFrame and display as enhanced table
                    df_diagnosis = pd.DataFrame(diagnosis_data)
                    
                    # Sort by claim date (most recent first)
                    df_diagnosis_sorted = df_diagnosis.sort_values('Claim Date', ascending=False, na_position='last')
                    
                    st.dataframe(
                        df_diagnosis_sorted, 
                        use_container_width=True, 
                        hide_index=True,
                        column_config={
                            "ICD-10 Code": st.column_config.TextColumn("ICD-10 Code", width="small"),
                            "Code Meaning": st.column_config.TextColumn("Medical Meaning", width="large"),
                            "Claim Date": st.column_config.DateColumn("Claim Date", width="small"),
                            "Service End Date": st.column_config.DateColumn("Service End Date", width="small"),
                            "Position": st.column_config.NumberColumn("Position", width="small"),
                            "Source Field": st.column_config.TextColumn("Source", width="small"),
                            "Record Path": st.column_config.TextColumn("Record Path", width="small"),
                            "Meaning Generated": st.column_config.TextColumn("Meaning Type", width="small")
                        }
                    )
                    
                    # Download button for the data
                    csv = df_diagnosis_sorted.to_csv(index=False)
                    st.info(f"📊 ICD-10 diagnosis data processed: {unique_codes} codes total, {codes_with_meanings} with LLM meanings")
                    
                    # Show code frequency analysis
                    with st.expander("📈 ICD-10 Code Frequency Analysis"):
                        code_counts = df_diagnosis['ICD-10 Code'].value_counts()
                        st.bar_chart(code_counts)
                        st.write("**Most Frequent Diagnosis Codes:**")
                        for code, count in code_counts.head(5).items():
                            meaning = next((item["Code Meaning"] for item in diagnosis_data if item["ICD-10 Code"] == code), "Unknown")
                            meaning_type = next((item["Meaning Generated"] for item in diagnosis_data if item["ICD-10 Code"] == code), "Unknown")
                            st.write(f"• **{code}** ({count}x): {meaning} [{meaning_type}]")
                else:
                    st.info("No ICD-10 diagnosis codes found in medical records")
            else:
                st.warning("No medical records available for ICD-10 analysis")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with med_tab2:
            st.markdown('<div class="code-table-container">', unsafe_allow_html=True)
            st.markdown("#### 🏥 Medical Service Codes with Dates and Meanings")
            
            if medical_records:
                # Prepare data for enhanced table display - INCLUDE ALL CODES
                service_data = []
                for record in medical_records:
                    claim_date = record.get("clm_rcvd_dt", "Unknown")
                    service_end_date = record.get("clm_line_srvc_end_dt", "Unknown")
                    service_code = record.get("hlth_srvc_cd", "")
                    record_path = record.get("data_path", "")
                    
                    if service_code:  # Include ALL service codes, not just those with meanings
                        # Use generated meaning if available, otherwise provide fallback
                        service_meaning = service_meanings.get(service_code, f"Medical service code {service_code}")
                        
                        service_data.append({
                            "Service Code": service_code,
                            "Service Meaning": service_meaning,
                            "Claim Date": claim_date,
                            "Service End Date": service_end_date,
                            "Record Path": record_path,
                            "Meaning Generated": "Yes" if service_code in service_meanings else "Fallback"
                        })
                
                if service_data:
                    # Display all codes count and statistics
                    unique_codes = len(set(item["Service Code"] for item in service_data))
                    codes_with_meanings = len([item for item in service_data if item["Meaning Generated"] == "Yes"])
                    codes_with_fallback = len([item for item in service_data if item["Meaning Generated"] == "Fallback"])
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📊 Unique Service Codes", unique_codes)
                    with col2:
                        st.metric("🧠 LLM Generated Meanings", codes_with_meanings)
                    with col3:
                        st.metric("🔄 Fallback Descriptions", codes_with_fallback)
                    
                    # Create DataFrame and display as enhanced table
                    df_service = pd.DataFrame(service_data)
                    
                    # Sort by claim date (most recent first)
                    df_service_sorted = df_service.sort_values('Claim Date', ascending=False, na_position='last')
                    
                    st.dataframe(
                        df_service_sorted, 
                        use_container_width=True, 
                        hide_index=True,
                        column_config={
                            "Service Code": st.column_config.TextColumn("Service Code", width="small"),
                            "Service Meaning": st.column_config.TextColumn("Service Description", width="large"),
                            "Claim Date": st.column_config.DateColumn("Claim Date", width="medium"),
                            "Service End Date": st.column_config.DateColumn("Service End Date", width="medium"),
                            "Record Path": st.column_config.TextColumn("Record Path", width="small"),
                            "Meaning Generated": st.column_config.TextColumn("Meaning Type", width="small")
                        }
                    )
                    
                    # Show service codes data
                    st.info(f"📊 Medical service codes processed: {unique_codes} codes total, {codes_with_meanings} with LLM meanings")
                    
                    # Show code frequency analysis
                    with st.expander("📈 Service Code Frequency Analysis"):
                        code_counts = df_service['Service Code'].value_counts()
                        st.bar_chart(code_counts)
                        st.write("**Most Frequent Service Codes:**")
                        for code, count in code_counts.head(5).items():
                            meaning = next((item["Service Meaning"] for item in service_data if item["Service Code"] == code), "Unknown")
                            meaning_type = next((item["Meaning Generated"] for item in service_data if item["Service Code"] == code), "Unknown")
                            st.write(f"• **{code}** ({count}x): {meaning} [{meaning_type}]")
                else:
                    st.info("No medical service codes found in records")
            else:
                st.warning("No medical records available for service code analysis")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="pharmacy-codes-section">', unsafe_allow_html=True)
        st.markdown("### 💊 Pharmacy Code Meanings Analysis")
        
        pharmacy_meanings = pharmacy_extraction.get("code_meanings", {})
        ndc_meanings = pharmacy_meanings.get("ndc_code_meanings", {})
        med_meanings = pharmacy_meanings.get("medication_meanings", {})
        pharmacy_records = pharmacy_extraction.get("ndc_records", [])
        
        # Pharmacy summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown('<div class="metric-summary-box">', unsafe_allow_html=True)
            st.metric("NDC Codes", len(ndc_meanings))
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-summary-box">', unsafe_allow_html=True)
            st.metric("Medications", len(med_meanings))
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-summary-box">', unsafe_allow_html=True)
            st.metric("Pharmacy Records", len(pharmacy_records))
            st.markdown('</div>', unsafe_allow_html=True)
        with col4:
            st.markdown('<div class="metric-summary-box">', unsafe_allow_html=True)
            batch_status = pharmacy_extraction.get("llm_call_status", "unknown")
            st.metric("Batch Status", batch_status)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Create sub-tabs for different pharmacy code types
        pharm_tab1, pharm_tab2 = st.tabs(["💊 NDC Codes", "💉 Medication Names"])
        
        with pharm_tab1:
            st.markdown('<div class="code-table-container">', unsafe_allow_html=True)
            st.markdown("#### 💊 NDC Codes with Fill Dates and Meanings")
            
            if pharmacy_records:
                # Prepare data for enhanced table display - INCLUDE ALL CODES
                ndc_data = []
                for record in pharmacy_records:
                    fill_date = record.get("rx_filled_dt", "Unknown")
                    ndc_code = record.get("ndc", "")
                    label_name = record.get("lbl_nm", "")
                    record_path = record.get("data_path", "")
                    
                    if ndc_code:  # Include ALL NDC codes, not just those with meanings
                        # Use generated meaning if available, otherwise provide fallback
                        ndc_meaning = ndc_meanings.get(ndc_code, f"NDC medication code {ndc_code}")
                        
                        ndc_data.append({
                            "NDC Code": ndc_code,
                            "NDC Meaning": ndc_meaning,
                            "Medication Name": label_name,
                            "Fill Date": fill_date,
                            "Record Path": record_path,
                            "Meaning Generated": "Yes" if ndc_code in ndc_meanings else "Fallback"
                        })
                
                if ndc_data:
                    # Display all codes count and statistics
                    unique_codes = len(set(item["NDC Code"] for item in ndc_data))
                    codes_with_meanings = len([item for item in ndc_data if item["Meaning Generated"] == "Yes"])
                    codes_with_fallback = len([item for item in ndc_data if item["Meaning Generated"] == "Fallback"])
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📊 Unique NDC Codes", unique_codes)
                    with col2:
                        st.metric("🧠 LLM Generated Meanings", codes_with_meanings)
                    with col3:
                        st.metric("🔄 Fallback Descriptions", codes_with_fallback)
                    
                    # Create DataFrame and display as enhanced table
                    df_ndc = pd.DataFrame(ndc_data)
                    
                    # Sort by fill date (most recent first)
                    df_ndc_sorted = df_ndc.sort_values('Fill Date', ascending=False, na_position='last')
                    
                    st.dataframe(
                        df_ndc_sorted, 
                        use_container_width=True, 
                        hide_index=True,
                        column_config={
                            "NDC Code": st.column_config.TextColumn("NDC Code", width="small"),
                            "NDC Meaning": st.column_config.TextColumn("NDC Description", width="large"),
                            "Medication Name": st.column_config.TextColumn("Medication", width="medium"),
                            "Fill Date": st.column_config.DateColumn("Fill Date", width="small"),
                            "Record Path": st.column_config.TextColumn("Record Path", width="small"),
                            "Meaning Generated": st.column_config.TextColumn("Meaning Type", width="small")
                        }
                    )
                    
                    # Show NDC codes data
                    st.info(f"📊 NDC codes processed: {unique_codes} codes total, {codes_with_meanings} with LLM meanings")
                    
                    # Show code frequency analysis
                    with st.expander("📈 NDC Code Frequency Analysis"):
                        code_counts = df_ndc['NDC Code'].value_counts()
                        st.bar_chart(code_counts)
                        st.write("**Most Frequent NDC Codes:**")
                        for code, count in code_counts.head(5).items():
                            meaning = next((item["NDC Meaning"] for item in ndc_data if item["NDC Code"] == code), "Unknown")
                            meaning_type = next((item["Meaning Generated"] for item in ndc_data if item["NDC Code"] == code), "Unknown")
                            st.write(f"• **{code}** ({count}x): {meaning} [{meaning_type}]")
                else:
                    st.info("No NDC codes found in pharmacy records")
            else:
                st.warning("No pharmacy records available for NDC analysis")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with pharm_tab2:
            st.markdown('<div class="code-table-container">', unsafe_allow_html=True)
            st.markdown("#### 💉 Medication Names with Fill Dates and Meanings")
            
            if pharmacy_records:
                # Prepare data for enhanced table display - INCLUDE ALL MEDICATIONS
                medication_data = []
                for record in pharmacy_records:
                    fill_date = record.get("rx_filled_dt", "Unknown")
                    med_name = record.get("lbl_nm", "")
                    ndc_code = record.get("ndc", "")
                    record_path = record.get("data_path", "")
                    
                    if med_name:  # Include ALL medication names, not just those with meanings
                        # Use generated meaning if available, otherwise provide fallback
                        med_meaning = med_meanings.get(med_name, f"Medication: {med_name}")
                        
                        medication_data.append({
                            "Medication Name": med_name,
                            "Medication Meaning": med_meaning,
                            "NDC Code": ndc_code,
                            "Fill Date": fill_date,
                            "Record Path": record_path,
                            "Meaning Generated": "Yes" if med_name in med_meanings else "Fallback"
                        })
                
                if medication_data:
                    # Display all medications count and statistics
                    unique_meds = len(set(item["Medication Name"] for item in medication_data))
                    meds_with_meanings = len([item for item in medication_data if item["Meaning Generated"] == "Yes"])
                    meds_with_fallback = len([item for item in medication_data if item["Meaning Generated"] == "Fallback"])
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📊 Unique Medications", unique_meds)
                    with col2:
                        st.metric("🧠 LLM Generated Meanings", meds_with_meanings)
                    with col3:
                        st.metric("🔄 Fallback Descriptions", meds_with_fallback)
                    
                    # Create DataFrame and display as enhanced table
                    df_medication = pd.DataFrame(medication_data)
                    
                    # Sort by fill date (most recent first)
                    df_medication_sorted = df_medication.sort_values('Fill Date', ascending=False, na_position='last')
                    
                    st.dataframe(
                        df_medication_sorted, 
                        use_container_width=True, 
                        hide_index=True,
                        column_config={
                            "Medication Name": st.column_config.TextColumn("Medication", width="medium"),
                            "Medication Meaning": st.column_config.TextColumn("Therapeutic Description", width="large"),
                            "NDC Code": st.column_config.TextColumn("NDC Code", width="small"),
                            "Fill Date": st.column_config.DateColumn("Fill Date", width="small"),
                            "Record Path": st.column_config.TextColumn("Record Path", width="small"),
                            "Meaning Generated": st.column_config.TextColumn("Meaning Type", width="small")
                        }
                    )
                    
                    # Show medications data
                    st.info(f"📊 Medication data processed: {unique_meds} medications total, {meds_with_meanings} with LLM meanings")
                    
                    # Show medication frequency analysis
                    with st.expander("📈 Medication Frequency Analysis"):
                        med_counts = df_medication['Medication Name'].value_counts()
                        st.bar_chart(med_counts)
                        st.write("**Most Frequent Medications:**")
                        for med, count in med_counts.head(5).items():
                            meaning = next((item["Medication Meaning"] for item in medication_data if item["Medication Name"] == med), "Unknown")
                            meaning_type = next((item["Meaning Generated"] for item in medication_data if item["Medication Name"] == med), "Unknown")
                            st.write(f"• **{med}** ({count}x): {meaning} [{meaning_type}]")
                else:
                    st.info("No medication names found in pharmacy records")
            else:
                st.warning("No pharmacy records available for medication analysis")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

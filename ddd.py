def display_batch_code_meanings_enhanced(results):
    """Enhanced batch processed code meanings in organized tabular format with proper subdivisions and PROVIDER FIELDS"""
    st.markdown("""
    <div class="batch-meanings-card">
        <h3>Claims Data Analysis</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Get extraction results
    medical_extraction = safe_get(results, 'structured_extractions', {}).get('medical', {})
    pharmacy_extraction = safe_get(results, 'structured_extractions', {}).get('pharmacy', {})
    
    # Create main tabs for Medical and Pharmacy
    tab1, tab2 = st.tabs(["Medical Code Meanings", "Pharmacy Code Meanings"])
    
    with tab1:
        st.markdown('<div class="medical-codes-section">', unsafe_allow_html=True)
        st.markdown("### Medical Code Meanings Analysis")
        
        medical_meanings = medical_extraction.get("code_meanings", {})
        service_meanings = medical_meanings.get("service_code_meanings", {})
        diagnosis_meanings = medical_meanings.get("diagnosis_code_meanings", {})
        medical_records = medical_extraction.get("hlth_srvc_records", [])
        
        # FIXED METRICS CALCULATION - Count unique codes from actual data
        unique_service_codes = set()
        unique_diagnosis_codes = set()
        total_medical_records = len(medical_records)
        
        # Count unique codes from medical records
        for record in medical_records:
            # Count service codes
            service_code = record.get("hlth_srvc_cd", "")
            if service_code:
                unique_service_codes.add(service_code)
            
            # Count diagnosis codes
            for diag in record.get("diagnosis_codes", []):
                code = diag.get("code", "")
                if code:
                    unique_diagnosis_codes.add(code)
        
        # Medical summary metrics with CORRECTED VALUES and PROPER STYLING
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'''
            <div class="metric-summary-box">
                <h3 style="margin: 0; color: #007bff; font-size: 2rem; font-weight: bold;">{len(unique_service_codes)}</h3>
                <p style="margin: 0; color: #6c757d; font-weight: 600;">Service Codes</p>
            </div>
            ''', unsafe_allow_html=True)
        with col2:
            st.markdown(f'''
            <div class="metric-summary-box">
                <h3 style="margin: 0; color: #28a745; font-size: 2rem; font-weight: bold;">{len(unique_diagnosis_codes)}</h3>
                <p style="margin: 0; color: #6c757d; font-weight: 600;">ICD-10 Codes</p>
            </div>
            ''', unsafe_allow_html=True)
        with col3:
            st.markdown(f'''
            <div class="metric-summary-box">
                <h3 style="margin: 0; color: #dc3545; font-size: 2rem; font-weight: bold;">{total_medical_records}</h3>
                <p style="margin: 0; color: #6c757d; font-weight: 600;">Medical Records</p>
            </div>
            ''', unsafe_allow_html=True)
        with col4:
            batch_status = medical_extraction.get("llm_call_status", "unknown")
            status_color = "#28a745" if batch_status in ["success", "completed"] else "#ffc107" if batch_status == "pending" else "#dc3545"
            st.markdown(f'''
            <div class="metric-summary-box">
                <h3 style="margin: 0; color: {status_color}; font-size: 1.5rem; font-weight: bold;">{batch_status.upper()}</h3>
                <p style="margin: 0; color: #6c757d; font-weight: 600;">Batch Status</p>
            </div>
            ''', unsafe_allow_html=True)
        
        # Create sub-tabs for different medical code types
        med_tab1, med_tab2 = st.tabs(["ICD-10 Diagnosis Codes", "Medical Service Codes"])
        
        with med_tab1:
            st.markdown('<div class="code-table-container">', unsafe_allow_html=True)
            st.markdown("#### ICD-10 Diagnosis Codes with Dates, Meanings, and Provider Information")
            
            if diagnosis_meanings and medical_records:
                # Prepare data for enhanced table display WITH PROVIDER FIELDS
                diagnosis_data = []
                for record in medical_records:
                    claim_date = record.get("clm_rcvd_dt", "Unknown")
                    record_path = record.get("data_path", "")
                    # NEW: Get provider information
                    billing_provider = record.get("billg_prov_nm", "Not Available")
                    billing_zip = record.get("billg_prov_zip_cd", "Not Available")
                    
                    for diag in record.get("diagnosis_codes", []):
                        code = diag.get("code", "")
                        if code in diagnosis_meanings:
                            diagnosis_data.append({
                                "ICD-10 Code": code,
                                "Code Meaning": diagnosis_meanings[code],
                                "Claim Date": claim_date,
                                "Billing Provider": billing_provider,
                                "Provider ZIP": billing_zip,
                                "Position": diag.get("position", ""),
                                "Source Field": diag.get("source", ""),
                                "Record Path": record_path
                            })
                
                if diagnosis_data:
                    # Display unique code count
                    unique_codes = len(set(item["ICD-10 Code"] for item in diagnosis_data))
                    st.info(f"**Unique ICD-10 Codes Found:** {unique_codes}")
                    
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
                            "Billing Provider": st.column_config.TextColumn("Billing Provider", width="medium"),
                            "Provider ZIP": st.column_config.TextColumn("Provider ZIP", width="small"),
                            "Position": st.column_config.NumberColumn("Position", width="small"),
                            "Source Field": st.column_config.TextColumn("Source", width="small"),
                            "Record Path": st.column_config.TextColumn("Record Path", width="small")
                        }
                    )
                    
                    st.info("ICD-10 diagnosis data with provider information processed successfully")
                    
                    # Show code frequency analysis
                    with st.expander("ICD-10 Code Frequency Analysis"):
                        code_counts = df_diagnosis['ICD-10 Code'].value_counts()
                        st.bar_chart(code_counts)
                        st.write("**Most Frequent Diagnosis Codes:**")
                        for code, count in code_counts.head(5).items():
                            meaning = diagnosis_meanings.get(code, "Unknown")
                            st.write(f"• **{code}** ({count}x): {meaning}")
                    
                    # Show provider analysis
                    with st.expander("Provider Analysis"):
                        provider_counts = df_diagnosis['Billing Provider'].value_counts()
                        st.write("**Most Frequent Billing Providers:**")
                        for provider, count in provider_counts.head(5).items():
                            if provider != "Not Available":
                                st.write(f"• **{provider}** ({count} claims)")
                        
                        zip_counts = df_diagnosis['Provider ZIP'].value_counts()
                        st.write("**Provider ZIP Distribution:**")
                        for zip_code, count in zip_counts.head(5).items():
                            if zip_code != "Not Available":
                                st.write(f"• **{zip_code}** ({count} claims)")
                else:
                    st.info("No ICD-10 diagnosis codes found in medical records")
            else:
                st.warning("No ICD-10 diagnosis code meanings available")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with med_tab2:
            st.markdown('<div class="code-table-container">', unsafe_allow_html=True)
            st.markdown("#### Medical Service Codes with Service End Dates and Meanings")
            
            if service_meanings and medical_records:
                # Prepare data for enhanced table display
                service_data = []
                for record in medical_records:
                    service_end_date = record.get("clm_line_srvc_end_dt", "Unknown")
                    service_code = record.get("hlth_srvc_cd", "")
                    record_path = record.get("data_path", "")
                    
                    if service_code and service_code in service_meanings:
                        service_data.append({
                            "Service Code": service_code,
                            "Service Meaning": service_meanings[service_code],
                            "Service End Date": service_end_date,
                            "Record Path": record_path
                        })
                
                if service_data:
                    # Display unique code count
                    unique_codes = len(set(item["Service Code"] for item in service_data))
                    st.info(f"**Unique Service Codes Found:** {unique_codes}")
                    
                    # Create DataFrame and display as enhanced table
                    df_service = pd.DataFrame(service_data)
                    
                    # Sort by service end date (most recent first)
                    df_service_sorted = df_service.sort_values('Service End Date', ascending=False, na_position='last')
                    
                    st.dataframe(
                        df_service_sorted, 
                        use_container_width=True, 
                        hide_index=True,
                        column_config={
                            "Service Code": st.column_config.TextColumn("Service Code", width="small"),
                            "Service Meaning": st.column_config.TextColumn("Service Description", width="large"),
                            "Service End Date": st.column_config.DateColumn("Service End Date", width="medium"),
                            "Record Path": st.column_config.TextColumn("Record Path", width="small")
                        }
                    )
                    
                    st.info("Medical service codes processed successfully")
                    
                    # Show code frequency analysis
                    with st.expander("Service Code Frequency Analysis"):
                        code_counts = df_service['Service Code'].value_counts()
                        st.bar_chart(code_counts)
                        st.write("**Most Frequent Service Codes:**")
                        for code, count in code_counts.head(5).items():
                            meaning = service_meanings.get(code, "Unknown")
                            st.write(f"• **{code}** ({count}x): {meaning}")
                else:
                    st.info("No medical service codes found in records")
            else:
                st.warning("No medical service code meanings available")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        st.markdown('<div class="pharmacy-codes-section">', unsafe_allow_html=True)
        st.markdown("### Pharmacy Code Meanings Analysis")
        
        pharmacy_meanings = pharmacy_extraction.get("code_meanings", {})
        ndc_meanings = pharmacy_meanings.get("ndc_code_meanings", {})
        med_meanings = pharmacy_meanings.get("medication_meanings", {})
        pharmacy_records = pharmacy_extraction.get("ndc_records", [])
        
        # FIXED METRICS CALCULATION - Count unique codes from actual pharmacy data
        unique_ndc_codes = set()
        unique_medications = set()
        unique_prescribing_providers = set()  # NEW: Count unique prescribing providers
        total_pharmacy_records = len(pharmacy_records)
        
        # Count unique codes from pharmacy records
        for record in pharmacy_records:
            # Count NDC codes
            ndc_code = record.get("ndc", "")
            if ndc_code:
                unique_ndc_codes.add(ndc_code)
            
            # Count medications
            med_name = record.get("lbl_nm", "")
            if med_name:
                unique_medications.add(med_name)
            
            # NEW: Count prescribing providers
            prescribing_provider = record.get("prscrb_prov_nm", "")
            if prescribing_provider and prescribing_provider != "Not Available":
                unique_prescribing_providers.add(prescribing_provider)
        
        # Pharmacy summary metrics with CORRECTED VALUES and UPDATED FOURTH METRIC
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.markdown(f'''
            <div class="metric-summary-box">
                <h3 style="margin: 0; color: #007bff; font-size: 2rem; font-weight: bold;">{len(unique_ndc_codes)}</h3>
                <p style="margin: 0; color: #6c757d; font-weight: 600;">NDC Codes</p>
            </div>
            ''', unsafe_allow_html=True)
        with col2:
            st.markdown(f'''
            <div class="metric-summary-box">
                <h3 style="margin: 0; color: #28a745; font-size: 2rem; font-weight: bold;">{len(unique_medications)}</h3>
                <p style="margin: 0; color: #6c757d; font-weight: 600;">Medications</p>
            </div>
            ''', unsafe_allow_html=True)
        with col3:
            st.markdown(f'''
            <div class="metric-summary-box">
                <h3 style="margin: 0; color: #dc3545; font-size: 2rem; font-weight: bold;">{total_pharmacy_records}</h3>
                <p style="margin: 0; color: #6c757d; font-weight: 600;">Pharmacy Records</p>
            </div>
            ''', unsafe_allow_html=True)
        with col4:
            # CHANGED: Show unique prescribing providers instead of batch status
            st.markdown(f'''
            <div class="metric-summary-box">
                <h3 style="margin: 0; color: #17a2b8; font-size: 2rem; font-weight: bold;">{len(unique_prescribing_providers)}</h3>
                <p style="margin: 0; color: #6c757d; font-weight: 600;">Prescribing Providers</p>
            </div>
            ''', unsafe_allow_html=True)
        
        # Create sub-tabs for different pharmacy code types
        pharm_tab1, pharm_tab2 = st.tabs(["NDC Codes", "Medication Names"])
        
        with pharm_tab1:
            st.markdown('<div class="code-table-container">', unsafe_allow_html=True)
            st.markdown("#### NDC Codes with Fill Dates and Meanings")
            
            if pharmacy_records:
                # Prepare data for enhanced table display
                ndc_data = []
                for record in pharmacy_records:
                    fill_date = record.get("rx_filled_dt", "Unknown")
                    ndc_code = record.get("ndc", "")
                    label_name = record.get("lbl_nm", "")
                    record_path = record.get("data_path", "")
                    
                    if ndc_code:  # Just check if NDC code exists
                        ndc_meaning = ndc_meanings.get(ndc_code, f"NDC code {ndc_code}")  # Use fallback if no meaning
                        ndc_data.append({
                            "NDC Code": ndc_code,
                            "NDC Meaning": ndc_meaning,
                            "Medication Name": label_name,
                            "Fill Date": fill_date,
                            "Record Path": record_path
                        })
                
                if ndc_data:
                    # Display unique code count
                    unique_codes = len(set(item["NDC Code"] for item in ndc_data))
                    st.info(f"**Unique NDC Codes Found:** {unique_codes}")
                    
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
                            "Record Path": st.column_config.TextColumn("Record Path", width="small")
                        }
                    )
                    
                    st.info("NDC codes data processed successfully")
                    
                    # Show code frequency analysis
                    with st.expander("NDC Code Frequency Analysis"):
                        code_counts = df_ndc['NDC Code'].value_counts()
                        st.bar_chart(code_counts)
                        st.write("**Most Frequent NDC Codes:**")
                        for code, count in code_counts.head(5).items():
                            meaning = ndc_meanings.get(code, f"NDC code {code}")
                            st.write(f"• **{code}** ({count}x): {meaning}")
                else:
                    st.info("No NDC codes found in pharmacy records")
            else:
                st.warning("No pharmacy records available for NDC analysis")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with pharm_tab2:
            st.markdown('<div class="code-table-container">', unsafe_allow_html=True)
            st.markdown("#### Medication Names with Fill Dates, Meanings, and Provider Information")
            
            if pharmacy_records:
                # Prepare data for enhanced table display WITH PROVIDER FIELDS
                medication_data = []
                for record in pharmacy_records:
                    fill_date = record.get("rx_filled_dt", "Unknown")
                    med_name = record.get("lbl_nm", "")
                    ndc_code = record.get("ndc", "")
                    record_path = record.get("data_path", "")
                    # NEW: Get provider information
                    billing_provider = record.get("billg_prov_nm", "Not Available")
                    prescribing_provider = record.get("prscrb_prov_nm", "Not Available")
                    
                    if med_name:  # Just check if medication name exists
                        med_meaning = med_meanings.get(med_name, f"Medication: {med_name}")  # Use fallback if no meaning
                        medication_data.append({
                            "Medication Name": med_name,
                            "Medication Meaning": med_meaning,
                            "NDC Code": ndc_code,
                            "Fill Date": fill_date,
                            "Billing Provider": billing_provider,
                            "Prescribing Provider": prescribing_provider,
                            "Record Path": record_path
                        })
                
                if medication_data:
                    # Display unique medication count
                    unique_meds = len(set(item["Medication Name"] for item in medication_data))
                    st.info(f"**Unique Medications Found:** {unique_meds}")
                    
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
                            "Billing Provider": st.column_config.TextColumn("Billing Provider", width="medium"),
                            "Prescribing Provider": st.column_config.TextColumn("Prescribing Provider", width="medium"),
                            "Record Path": st.column_config.TextColumn("Record Path", width="small")
                        }
                    )
                    
                    st.info("Medication data with provider information processed successfully")
                    
                    # Show medication frequency analysis
                    with st.expander("Medication Frequency Analysis"):
                        med_counts = df_medication['Medication Name'].value_counts()
                        st.bar_chart(med_counts)
                        st.write("**Most Frequent Medications:**")
                        for med, count in med_counts.head(5).items():
                            meaning = med_meanings.get(med, f"Medication: {med}")
                            st.write(f"• **{med}** ({count}x): {meaning}")
                    
                    # NEW: Show provider analysis
                    with st.expander("Provider Analysis"):
                        billing_provider_counts = df_medication['Billing Provider'].value_counts()
                        st.write("**Most Frequent Billing Providers:**")
                        for provider, count in billing_provider_counts.head(5).items():
                            if provider != "Not Available":
                                st.write(f"• **{provider}** ({count} prescriptions)")
                        
                        prescribing_provider_counts = df_medication['Prescribing Provider'].value_counts()
                        st.write("**Most Frequent Prescribing Providers:**")
                        for provider, count in prescribing_provider_counts.head(5).items():
                            if provider != "Not Available":
                                st.write(f"• **{provider}** ({count} prescriptions)")
                else:
                    st.info("No medication names found in pharmacy records")
            else:
                st.warning("No pharmacy records available for medication analysis")
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)

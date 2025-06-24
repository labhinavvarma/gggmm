def _extract_year_from_question(self, question: str) -> Optional[int]:
    match = re.search(r'\b(19|20)\d{2}\b', question)
    if match:
        return int(match.group(0))
    return None


def _extract_records_by_year(self, data: Any, year: Optional[int]) -> List[Dict[str, Any]]:
    matched_records = []

    def match_by_year(entry):
        for key in ['date', 'service_date', 'claim_date']:
            if key in entry:
                try:
                    entry_year = int(entry[key][:4])
                    if entry_year == year:
                        return True
                except:
                    continue
        return False

    def recurse(d):
        if isinstance(d, list):
            for item in d:
                recurse(item)
        elif isinstance(d, dict):
            if match_by_year(d):
                matched_records.append(d)
            for v in d.values():
                recurse(v)

    if year:
        recurse(data)
        return matched_records
    else:
        return []


def _count_medical_claims(self, deident_medical: Dict, raw_medical: Dict, year: Optional[int] = None) -> Dict[str, Any]:
    result = {
        'total_claims': 0,
        'deident_records': 0,
        'raw_records': 0,
        'service_entries': 0,
        'details': []
    }

    try:
        if deident_medical and not deident_medical.get('error'):
            medical_data = deident_medical.get('medical_data', {})
            if year:
                records = self._extract_records_by_year(medical_data, year)
                result['deident_records'] = len(records)
                result['details'].append(f"Deidentified medical records in {year}: {len(records)}")
            else:
                count = self._recursive_count_records(medical_data)
                result['deident_records'] = count
                result['details'].append(f"Deidentified medical records: {count}")

        if raw_medical and not raw_medical.get('error'):
            if year:
                records = self._extract_records_by_year(raw_medical, year)
                result['raw_records'] = len(records)
                result['details'].append(f"Raw medical response records in {year}: {len(records)}")
            else:
                count = self._recursive_count_records(raw_medical)
                result['raw_records'] = count
                result['details'].append(f"Raw medical response records: {count}")

        result['total_claims'] = max(result['deident_records'], result['raw_records'])

        if result['total_claims'] == 0:
            result['details'].append("No medical claims found in the data")

    except Exception as e:
        result['details'].append(f"Error counting medical claims: {str(e)}")

    return result


def _try_direct_data_analysis(self, user_question: str) -> Optional[str]:
    if not self.rag_knowledge_base:
        return None

    try:
        question_lower = user_question.lower()
        medical_data = self.rag_knowledge_base.get('deidentified_medical_data', {})
        pharmacy_data = self.rag_knowledge_base.get('deidentified_pharmacy_data', {})
        entities = self.rag_knowledge_base.get('entity_extraction_results', {})
        raw_responses = self.rag_knowledge_base.get('raw_api_responses', {})

        year = self._extract_year_from_question(user_question)

        if any(phrase in question_lower for phrase in ["medical claims", "number of medical", "how many medical", "count medical"]):
            medical_count = self._count_medical_claims(medical_data, raw_responses.get('medical', {}), year)
            return f"""\U0001F4CA **Medical Claims Analysis (RAG Mode):**

**Total Medical Claims Found{f' in {year}' if year else ''}:** {medical_count['total_claims']}

**Detailed Breakdown:**
- Deidentified medical records: {medical_count['deident_records']}
- Raw medical response records: {medical_count['raw_records']}
- Service entries identified: {medical_count['service_entries']}

**Analysis Details:**
{chr(10).join(medical_count['details'])}

*Source: RAG analysis of deidentified MCP server data*"""

        return None

    except Exception as e:
        logger.error(f"Error in direct data analysis: {e}")
        return f"I encountered an error analyzing the data: {str(e)}"

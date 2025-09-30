import React, { useState, useMemo } from 'react';
import { ChevronDown, ChevronUp, Calendar, Pill, FileText, Building2 } from 'lucide-react';

// Sample data structure - replace with your API call
const sampleResults = {
  structured_extractions: {
    medical: {
      hlth_srvc_records: [
        {
          clm_rcvd_dt: "2024-03-15",
          billg_prov_nm: "City Medical Center",
          billg_prov_zip_cd: "30301",
          data_path: "/medical/claim_001",
          hlth_srvc_cd: "99213",
          clm_line_srvc_end_dt: "2024-03-15",
          diagnosis_codes: [
            { code: "E11.9", position: 1, source: "primary" },
            { code: "I10", position: 2, source: "secondary" }
          ]
        },
        {
          clm_rcvd_dt: "2024-02-10",
          billg_prov_nm: "Northside Clinic",
          billg_prov_zip_cd: "30305",
          data_path: "/medical/claim_002",
          hlth_srvc_cd: "99214",
          clm_line_srvc_end_dt: "2024-02-10",
          diagnosis_codes: [
            { code: "J45.909", position: 1, source: "primary" }
          ]
        }
      ],
      code_meanings: {
        service_code_meanings: {
          "99213": "Office visit, established patient, 20-29 minutes",
          "99214": "Office visit, established patient, 30-39 minutes"
        },
        diagnosis_code_meanings: {
          "E11.9": "Type 2 diabetes mellitus without complications",
          "I10": "Essential (primary) hypertension",
          "J45.909": "Unspecified asthma, uncomplicated"
        }
      }
    },
    pharmacy: {
      ndc_records: [
        {
          rx_filled_dt: "2024-03-20",
          ndc: "00002-7510-01",
          lbl_nm: "METFORMIN HCL 500MG",
          billg_prov_nm: "CVS Pharmacy #1234",
          prscrbg_prov_nm: "Dr. Sarah Johnson",
          data_path: "/pharmacy/rx_001"
        },
        {
          rx_filled_dt: "2024-02-15",
          ndc: "00093-0058-01",
          lbl_nm: "LISINOPRIL 10MG",
          billg_prov_nm: "Walgreens #5678",
          prscrbg_prov_nm: "Dr. Michael Chen",
          data_path: "/pharmacy/rx_002"
        }
      ],
      code_meanings: {
        ndc_code_meanings: {
          "00002-7510-01": "Metformin hydrochloride 500mg tablet",
          "00093-0058-01": "Lisinopril 10mg tablet"
        },
        medication_meanings: {
          "METFORMIN HCL 500MG": "Oral diabetes medication to control blood sugar",
          "LISINOPRIL 10MG": "ACE inhibitor for high blood pressure"
        }
      }
    }
  }
};

const DataTable = ({ data, columns, sortKey = null }) => {
  const [sortField, setSortField] = useState(sortKey || columns[0].key);
  const [sortDirection, setSortDirection] = useState('desc');

  const sortedData = useMemo(() => {
    const sorted = [...data].sort((a, b) => {
      const aVal = a[sortField] || '';
      const bVal = b[sortField] || '';
      
      if (sortDirection === 'asc') {
        return aVal > bVal ? 1 : -1;
      }
      return aVal < bVal ? 1 : -1;
    });
    return sorted;
  }, [data, sortField, sortDirection]);

  const handleSort = (field) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc');
    } else {
      setSortField(field);
      setSortDirection('desc');
    }
  };

  return (
    <div className="overflow-x-auto rounded-lg border border-gray-200 shadow-sm">
      <table className="min-w-full divide-y divide-gray-200">
        <thead className="bg-gray-50">
          <tr>
            {columns.map((col) => (
              <th
                key={col.key}
                onClick={() => handleSort(col.key)}
                className="px-6 py-3 text-left text-xs font-medium text-gray-700 uppercase tracking-wider cursor-pointer hover:bg-gray-100 transition-colors"
              >
                <div className="flex items-center gap-2">
                  {col.label}
                  {sortField === col.key && (
                    sortDirection === 'asc' ? <ChevronUp size={14} /> : <ChevronDown size={14} />
                  )}
                </div>
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="bg-white divide-y divide-gray-200">
          {sortedData.map((row, idx) => (
            <tr key={idx} className="hover:bg-gray-50 transition-colors">
              {columns.map((col) => (
                <td key={col.key} className="px-6 py-4 text-sm text-gray-900">
                  {col.render ? col.render(row[col.key], row) : row[col.key] || 'N/A'}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default function HealthClaimsTables() {
  const [activeTab, setActiveTab] = useState('medical');
  
  // Extract data from results
  const medicalExtraction = sampleResults.structured_extractions.medical;
  const pharmacyExtraction = sampleResults.structured_extractions.pharmacy;
  
  const medicalRecords = medicalExtraction.hlth_srvc_records;
  const pharmacyRecords = pharmacyExtraction.ndc_records;
  
  const serviceMeanings = medicalExtraction.code_meanings.service_code_meanings;
  const diagnosisMeanings = medicalExtraction.code_meanings.diagnosis_code_meanings;
  const ndcMeanings = pharmacyExtraction.code_meanings.ndc_code_meanings;
  const medMeanings = pharmacyExtraction.code_meanings.medication_meanings;

  // Process diagnosis data
  const diagnosisData = useMemo(() => {
    const data = [];
    medicalRecords.forEach(record => {
      record.diagnosis_codes?.forEach(diag => {
        if (diagnosisMeanings[diag.code]) {
          data.push({
            code: diag.code,
            meaning: diagnosisMeanings[diag.code],
            claimDate: record.clm_rcvd_dt,
            billingProvider: record.billg_prov_nm,
            providerZip: record.billg_prov_zip_cd,
            position: diag.position,
            source: diag.source,
            recordPath: record.data_path
          });
        }
      });
    });
    return data;
  }, [medicalRecords, diagnosisMeanings]);

  // Process service data
  const serviceData = useMemo(() => {
    const data = [];
    medicalRecords.forEach(record => {
      if (record.hlth_srvc_cd && serviceMeanings[record.hlth_srvc_cd]) {
        data.push({
          serviceCode: record.hlth_srvc_cd,
          serviceMeaning: serviceMeanings[record.hlth_srvc_cd],
          serviceEndDate: record.clm_line_srvc_end_dt,
          recordPath: record.data_path
        });
      }
    });
    return data;
  }, [medicalRecords, serviceMeanings]);

  // Process NDC data
  const ndcData = useMemo(() => {
    return pharmacyRecords.map(record => ({
      ndcCode: record.ndc,
      ndcMeaning: ndcMeanings[record.ndc] || `NDC code ${record.ndc}`,
      medicationName: record.lbl_nm,
      fillDate: record.rx_filled_dt,
      recordPath: record.data_path
    }));
  }, [pharmacyRecords, ndcMeanings]);

  // Process medication data
  const medicationData = useMemo(() => {
    return pharmacyRecords.map(record => ({
      medicationName: record.lbl_nm,
      medicationMeaning: medMeanings[record.lbl_nm] || `Medication: ${record.lbl_nm}`,
      ndcCode: record.ndc,
      fillDate: record.rx_filled_dt,
      billingProvider: record.billg_prov_nm || 'Not Available',
      prescribingProvider: record.prscrbg_prov_nm || 'Not Available',
      recordPath: record.data_path
    }));
  }, [pharmacyRecords, medMeanings]);

  // Column definitions
  const diagnosisColumns = [
    { key: 'code', label: 'ICD-10 Code' },
    { key: 'meaning', label: 'Code Meaning' },
    { key: 'claimDate', label: 'Claim Date' },
    { key: 'billingProvider', label: 'Billing Provider' },
    { key: 'providerZip', label: 'ZIP' },
    { key: 'position', label: 'Pos' },
  ];

  const serviceColumns = [
    { key: 'serviceCode', label: 'Service Code' },
    { key: 'serviceMeaning', label: 'Service Meaning' },
    { key: 'serviceEndDate', label: 'Service End Date' },
  ];

  const ndcColumns = [
    { key: 'ndcCode', label: 'NDC Code' },
    { key: 'ndcMeaning', label: 'NDC Meaning' },
    { key: 'medicationName', label: 'Medication Name' },
    { key: 'fillDate', label: 'Fill Date' },
  ];

  const medicationColumns = [
    { key: 'medicationName', label: 'Medication' },
    { key: 'medicationMeaning', label: 'Therapeutic Description' },
    { key: 'ndcCode', label: 'NDC Code' },
    { key: 'fillDate', label: 'Fill Date' },
    { key: 'billingProvider', label: 'Billing Provider' },
    { key: 'prescribingProvider', label: 'Prescribing Provider' },
  ];

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-indigo-50 p-8">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-4xl font-bold text-gray-900 mb-8">
          Health Claims Data Tables
        </h1>

        {/* Tabs */}
        <div className="mb-6 border-b border-gray-200">
          <nav className="flex gap-4">
            <button
              onClick={() => setActiveTab('medical')}
              className={`px-6 py-3 font-semibold border-b-2 transition-colors ${
                activeTab === 'medical'
                  ? 'border-blue-600 text-blue-600'
                  : 'border-transparent text-gray-600 hover:text-gray-800'
              }`}
            >
              <div className="flex items-center gap-2">
                <FileText size={20} />
                Medical Tables
              </div>
            </button>
            <button
              onClick={() => setActiveTab('pharmacy')}
              className={`px-6 py-3 font-semibold border-b-2 transition-colors ${
                activeTab === 'pharmacy'
                  ? 'border-blue-600 text-blue-600'
                  : 'border-transparent text-gray-600 hover:text-gray-800'
              }`}
            >
              <div className="flex items-center gap-2">
                <Pill size={20} />
                Pharmacy Tables
              </div>
            </button>
          </nav>
        </div>

        {/* Medical Tables */}
        {activeTab === 'medical' && (
          <div className="space-y-8">
            <div className="bg-white rounded-lg p-6 shadow-lg">
              <h2 className="text-2xl font-semibold text-gray-800 mb-4 flex items-center gap-2">
                <FileText size={24} className="text-blue-600" />
                ICD-10 Diagnosis Codes
              </h2>
              <p className="text-sm text-gray-600 mb-4">
                {diagnosisData.length} diagnosis records found
              </p>
              <DataTable 
                data={diagnosisData} 
                columns={diagnosisColumns}
                sortKey="claimDate"
              />
            </div>

            <div className="bg-white rounded-lg p-6 shadow-lg">
              <h2 className="text-2xl font-semibold text-gray-800 mb-4 flex items-center gap-2">
                <Building2 size={24} className="text-green-600" />
                Medical Service Codes
              </h2>
              <p className="text-sm text-gray-600 mb-4">
                {serviceData.length} service records found
              </p>
              <DataTable 
                data={serviceData} 
                columns={serviceColumns}
                sortKey="serviceEndDate"
              />
            </div>
          </div>
        )}

        {/* Pharmacy Tables */}
        {activeTab === 'pharmacy' && (
          <div className="space-y-8">
            <div className="bg-white rounded-lg p-6 shadow-lg">
              <h2 className="text-2xl font-semibold text-gray-800 mb-4 flex items-center gap-2">
                <Pill size={24} className="text-purple-600" />
                NDC Codes
              </h2>
              <p className="text-sm text-gray-600 mb-4">
                {ndcData.length} NDC records found
              </p>
              <DataTable 
                data={ndcData} 
                columns={ndcColumns}
                sortKey="fillDate"
              />
            </div>

            <div className="bg-white rounded-lg p-6 shadow-lg">
              <h2 className="text-2xl font-semibold text-gray-800 mb-4 flex items-center gap-2">
                <Calendar size={24} className="text-indigo-600" />
                Medication Names
              </h2>
              <p className="text-sm text-gray-600 mb-4">
                {medicationData.length} medication records found
              </p>
              <DataTable 
                data={medicationData} 
                columns={medicationColumns}
                sortKey="fillDate"
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

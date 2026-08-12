# Incident Response Policy

> **Sample document.** This is fictional content created for the Compliance Copilot demo project. It does not describe the policies of any real company.

**Document owner:** Information Security Team
**Effective date:** 2025-01-01
**Applies to:** All employees and systems.

## 1. Purpose

This policy defines how the company detects, responds to, and recovers from security incidents, including data breaches.

## 2. Incident Severity Levels

- **Sev 1 (Critical):** Confirmed data breach involving customer personal data, or a system outage affecting all customers.
- **Sev 2 (High):** Suspected data breach, or an outage affecting a significant subset of customers.
- **Sev 3 (Medium):** Contained security issue with no evidence of data exposure, e.g., a blocked intrusion attempt.
- **Sev 4 (Low):** Minor policy violations or isolated, low-impact issues.

## 3. Response Timeline

### 3.1 Detection and Triage

Any suspected incident must be reported to the security team within 1 hour of discovery. The on-call security engineer triages and assigns a severity level within 2 hours of the report.

### 3.2 Containment

For Sev 1 and Sev 2 incidents, containment actions (e.g., revoking credentials, isolating affected systems) must begin within 4 hours of triage.

### 3.3 Customer and Regulatory Notification

If a Sev 1 incident is confirmed to involve exposure of customer personal data, affected customers must be notified within 72 hours of confirmation, consistent with applicable data protection regulations. Regulatory notification timelines follow the applicable jurisdiction's requirements, coordinated by Legal.

### 3.4 Post-Incident Review

A post-incident review (blameless postmortem) must be completed within 10 business days of resolution for all Sev 1 and Sev 2 incidents, documenting root cause, impact, and remediation actions.

## 4. Roles and Responsibilities

- **Incident Commander:** Coordinates the response, communication, and decision-making during an active incident.
- **Security On-Call:** First responder for triage and containment.
- **Legal/Compliance:** Determines notification obligations and manages regulatory communication.
- **Communications:** Manages internal and external messaging during the incident.

## 5. Evidence Preservation

Logs, system snapshots, and other evidence related to a Sev 1 or Sev 2 incident must be preserved for at least 5 years and must not be altered or deleted, even if this conflicts with standard log retention periods.

## 6. Tabletop Exercises

The security team runs incident response tabletop exercises at least twice per year to test readiness against realistic breach scenarios.

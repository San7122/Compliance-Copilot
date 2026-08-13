# Information Security Policy

> **Sample document.** This is fictional content created for the Compliance Copilot demo project. It does not describe the policies of any real company.

**Document owner:** Information Security Team
**Effective date:** 2025-01-01
**Applies to:** All employees, contractors, and third parties with access to company systems.

## 1. Purpose

This policy establishes baseline security requirements for protecting company and customer information across systems, applications, and infrastructure.

## 2. Access Control

### 2.1 Authentication

All employee accounts must use single sign-on (SSO) with multi-factor authentication (MFA) enabled. Passwords, where still required, must be at least 12 characters and are rotated only if compromise is suspected (not on a fixed schedule).

### 2.2 Least Privilege

Access to systems and data is granted on a least-privilege, need-to-know basis. Access requests must be approved by the requester's manager and the relevant system owner.

### 2.3 Access Reviews

Access rights for all production systems are reviewed quarterly. Any access not actively used in the prior 90 days is automatically revoked pending re-justification.

## 3. Data Encryption

- Data in transit must use TLS 1.2 or higher.
- Data at rest containing customer or employee personal data must be encrypted using AES-256 or an equivalent standard.
- Encryption keys are managed centrally and rotated at least annually.

## 4. Endpoint Security

All company-issued laptops must run approved endpoint detection and response (EDR) software, have full-disk encryption enabled, and receive security patches within 14 days of release for critical vulnerabilities.

## 5. Third-Party and Vendor Security

Vendors that process company or customer data must complete a security review before onboarding and must be re-assessed annually. Vendor contracts must include data protection and breach notification clauses.

## 6. Vulnerability Management

Critical vulnerabilities identified in production systems must be remediated within 7 days. High-severity vulnerabilities must be remediated within 30 days. Vulnerability scans are run weekly on all internet-facing systems.

## 7. Security Awareness Training

All employees must complete security awareness training within their first 30 days of employment and annually thereafter. Phishing simulation exercises are run at least quarterly.

## 8. Reporting Security Concerns

Suspected security issues should be reported immediately to security@example.com or via the internal security hotline. See the Incident Response Policy for what happens after a report is made.

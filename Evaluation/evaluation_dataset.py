"""Evaluation dataset for HR Policy RAG pipeline - retriever and generation phases."""

from dataclasses import dataclass
from typing import List, Dict, Any


@dataclass
class EvaluationSample:
    """Single evaluation sample with user_input, expected answer, and ground truth contexts."""
    user_input: str
    reference: str
    ground_truth_contexts: List[str]
    category: str
    difficulty: str  # "easy", "medium", "hard"


HR_POLICY_EVAL_DATASET: List[EvaluationSample] = [
    # ==================== BENEFITS & PERKS ====================
    EvaluationSample(
        user_input="What is the parental leave policy for new parents?",
        reference="Resilience X provides 20 weeks of fully paid parental leave for birthing parents and 12 weeks for non-birthing parents, available within the first 12 months of birth, adoption, or foster placement.",
        ground_truth_contexts=[
            "Content/benefits-pay-perks/benefits-perks/parental-leave.md",
            "Content/benefits-pay-perks/benefits-perks/benefits.md"
        ],
        category="benefits",
        difficulty="easy"
    ),
    EvaluationSample(
        user_input="How many weeks of paid parental leave do non-birthing parents get?",
        reference="Non-birthing parents receive 12 weeks of fully paid parental leave.",
        ground_truth_contexts=[
            "Content/benefits-pay-perks/benefits-perks/parental-leave.md"
        ],
        category="benefits",
        difficulty="easy"
    ),
    EvaluationSample(
        user_input="What mental health benefits are available to employees?",
        reference="Employees have access to mental health benefits including therapy sessions, Employee Assistance Program (EAP), and mental health days as part of the benefits package.",
        ground_truth_contexts=[
            "Content/benefits-pay-perks/benefits-perks/mental-health/",
            "Content/benefits-pay-perks/benefits-perks/benefits.md"
        ],
        category="benefits",
        difficulty="medium"
    ),
    EvaluationSample(
        user_input="What is the policy for spending company money on team events?",
        reference="Team events have a budget of $50/person/quarter for virtual events and $100/person/quarter for in-person events, with pre-approval required for amounts exceeding these limits.",
        ground_truth_contexts=[
            "Content/benefits-pay-perks/benefits-perks/spending-company-money.md"
        ],
        category="benefits",
        difficulty="medium"
    ),
    EvaluationSample(
        user_input="Can I take a leave of absence for personal reasons?",
        reference="Yes, Resilience X offers personal leaves of absence up to 30 days unpaid with manager approval, and extended leaves up to 90 days with HR approval for qualifying circumstances.",
        ground_truth_contexts=[
            "Content/benefits-pay-perks/benefits-perks/leave-of-absence.md"
        ],
        category="benefits",
        difficulty="hard"
    ),

    # ==================== TIME OFF & LEAVE ====================
    EvaluationSample(
        user_input="How many vacation days do employees get per year?",
        reference="Employees receive 20 days of paid vacation per year, accrued monthly, with a maximum carryover of 5 days to the next year.",
        ground_truth_contexts=[
            "Content/benefits-pay-perks/benefits-perks/time-off/",
            "Content/benefits-pay-perks/benefits-perks/benefits.md"
        ],
        category="time_off",
        difficulty="easy"
    ),
    EvaluationSample(
        user_input="What holidays does the company observe?",
        reference="The company observes 11 paid holidays including New Year's Day, Martin Luther King Jr. Day, Presidents' Day, Memorial Day, Juneteenth, Independence Day, Labor Day, Thanksgiving, Day after Thanksgiving, Christmas Eve, and Christmas Day.",
        ground_truth_contexts=[
            "Content/benefits-pay-perks/benefits-perks/time-off/",
            "Content/benefits-pay-perks/benefits-perks/benefits.md"
        ],
        category="time_off",
        difficulty="easy"
    ),
    EvaluationSample(
        user_input="Can I donate my unused PTO to a colleague?",
        reference="Yes, employees can donate up to 5 days of accrued vacation per year to colleagues facing medical emergencies through the PTO donation program.",
        ground_truth_contexts=[
            "Content/benefits-pay-perks/benefits-perks/time-off/",
            "Content/benefits-pay-perks/benefits-perks/leave-of-absence.md"
        ],
        category="time_off",
        difficulty="medium"
    ),

    # ==================== COMPENSATION & EXPENSES ====================
    EvaluationSample(
        user_input="How often are performance reviews conducted?",
        reference="Performance reviews are conducted semi-annually (twice per year) with a mid-year check-in and annual review cycle.",
        ground_truth_contexts=[
            "Content/benefits-pay-perks/pay-expenses/compensation/",
            "Content/departments/people-talent/"
        ],
        category="compensation",
        difficulty="easy"
    ),
    EvaluationSample(
        user_input="What is the expense reimbursement policy for travel?",
        reference="Travel expenses are reimbursed per diem rates: $75/day for meals, actual receipts for lodging/transport, with pre-approval required for trips over $500.",
        ground_truth_contexts=[
            "Content/benefits-pay-perks/pay-expenses/expenses/",
            "Content/benefits-pay-perks/benefits-perks/travel/"
        ],
        category="compensation",
        difficulty="medium"
    ),
    EvaluationSample(
        user_input="What is the 401(k) matching policy?",
        reference="The company matches 100% of employee contributions up to 4% of eligible compensation, with immediate vesting.",
        ground_truth_contexts=[
            "Content/benefits-pay-perks/pay-expenses/compensation/",
            "Content/benefits-pay-perks/benefits-perks/benefits.md"
        ],
        category="compensation",
        difficulty="easy"
    ),

    # ==================== COMPANY POLICIES ====================
    EvaluationSample(
        user_input="What is the respectful workplace policy?",
        reference="Resilience X maintains a zero-tolerance policy for harassment, discrimination, and retaliation. All employees must complete annual respectful workplace training.",
        ground_truth_contexts=[
            "Content/company-info-and-process/policies/respectful-workplace-policy.md"
        ],
        category="policies",
        difficulty="easy"
    ),
    EvaluationSample(
        user_input="What is the policy on gender diversity and inclusion?",
        reference="The company commits to gender diversity through inclusive hiring practices, pay equity audits, employee resource groups, and transparent promotion criteria.",
        ground_truth_contexts=[
            "Content/company-info-and-process/gender-diversity.md",
            "Content/company-info-and-process/equality-of-opportunity.md"
        ],
        category="policies",
        difficulty="medium"
    ),
    EvaluationSample(
        user_input="How does the company handle personal pronouns in the workplace?",
        reference="Employees are encouraged to share their pronouns in email signatures, Slack profiles, and introductions. Misgendering repeatedly after correction may violate the respectful workplace policy.",
        ground_truth_contexts=[
            "Content/company-info-and-process/personal-pronouns.md"
        ],
        category="policies",
        difficulty="medium"
    ),
    EvaluationSample(
        user_input="What is the remote work policy?",
        reference="Resilience X operates as a remote-first company. Employees can work from anywhere with manager alignment on time zone overlap for collaboration (minimum 4 hours overlap with team core hours).",
        ground_truth_contexts=[
            "Content/company-info-and-process/remote/"
        ],
        category="policies",
        difficulty="easy"
    ),

    # ==================== ONBOARDING & PEOPLE OPERATIONS ====================
    EvaluationSample(
        user_input="What does the onboarding process look like for new hires?",
        reference="New hires complete a 2-week structured onboarding including orientation, team introductions, systems setup, policy training, and a 30-60-90 day plan with their manager.",
        ground_truth_contexts=[
            "Content/company-info-and-process/onboarding/",
            "Content/departments/people-talent/"
        ],
        category="onboarding",
        difficulty="medium"
    ),
    EvaluationSample(
        user_input="How does the mentorship program work?",
        reference="The mentorship program pairs employees with mentors for 6-month cycles, with monthly check-ins, skill development goals, and optional reverse mentoring.",
        ground_truth_contexts=[
            "Content/company-info-and-process/mentorship/"
        ],
        category="onboarding",
        difficulty="medium"
    ),

    # ==================== LEGAL & COMPLIANCE ====================
    EvaluationSample(
        user_input="What is the process for reporting workplace concerns?",
        reference="Employees can report concerns through their manager, HR Business Partner, anonymous ethics hotline, or the online reporting portal. All reports are investigated within 5 business days.",
        ground_truth_contexts=[
            "Content/departments/legal/process/",
            "Content/company-info-and-process/policies/respectful-workplace-policy.md"
        ],
        category="legal",
        difficulty="medium"
    ),
    EvaluationSample(
        user_input="What data privacy protections apply to employee information?",
        reference="Employee data is protected under GDPR, CCPA, and local privacy laws. Access is limited to HR and direct managers on a need-to-know basis. Employees can request data access/deletion.",
        ground_truth_contexts=[
            "Content/departments/legal/",
            "Content/departments/legal/process/"
        ],
        category="legal",
        difficulty="hard"
    ),

    # ==================== EDGE CASES / TRICK user_inputS ====================
    EvaluationSample(
        user_input="What is the policy on cryptocurrency compensation?",
        reference="The company does not currently offer cryptocurrency as a form of compensation. All compensation is paid in local currency via direct deposit.",
        ground_truth_contexts=[],
        category="edge_case",
        difficulty="hard"
    ),
    EvaluationSample(
        user_input="Can I bring my pet to the office?",
        reference="Since Resilience X is a remote-first company, there are no physical offices to bring pets to. For co-working space visits, local pet policies apply.",
        ground_truth_contexts=[
            "Content/company-info-and-process/remote/"
        ],
        category="edge_case",
        difficulty="medium"
    ),
    EvaluationSample(
        user_input="What is the sabbatical policy after 5 years?",
        reference="The current policy documents do not mention a formal sabbatical program. Extended leaves of absence up to 90 days may be available for qualifying circumstances.",
        ground_truth_contexts=[
            "Content/benefits-pay-perks/benefits-perks/leave-of-absence.md"
        ],
        category="edge_case",
        difficulty="hard"
    ),
]

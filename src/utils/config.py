"""Configuration constants for candidate ranking and scoring."""

W_JD_MATCH: float = 0.40
W_CAREER: float = 0.30
W_BEHAVIORAL: float = 0.20
W_LOCATION: float = 0.10

CONSULTING_PENALTY: float = 0.50
RESEARCH_PENALTY: float = 0.40
HONEYPOT_PENALTY: float = 0.30

RECENCY_HALFLIFE_DAYS: int = 60

PREFERRED_CITIES: set[str] = {
    "Bengaluru",
    "Hyderabad",
    "Pune",
    "Chennai",
    "Gurugram",
    "Noida",
    "Mumbai",
    "Delhi",
    "NCR",
    "Ahmedabad",
    "Kolkata",
    "Kochi",
    "Coimbatore",
    "Chandigarh",
}

GOOD_INDUSTRIES: set[str] = {
    "AI/ML",
    "SaaS",
    "FinTech",
    "EdTech",
    "Health Tech",
}

BAD_INDUSTRIES: set[str] = {
    "IT Services",
    "Consulting",
    "Outsourcing",
}

CONSULTING_FIRMS: set[str] = {
    "TCS",
    "Infosys",
    "Wipro",
    "Accenture",
    "Cognizant",
    "Capgemini",
}

NOTICE_PERIOD_THRESHOLDS: dict[str, int] = {
    "immediate_days": 0,
    "preferred_max_days": 30,
    "acceptable_max_days": 60,
    "high_risk_min_days": 90,
}

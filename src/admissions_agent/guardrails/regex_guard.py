import re

from admissions_agent.guardrails.result import GuardrailResult

GUARD_NAME = "regex"
MAX_INPUT_LENGTH = 500

BLOCKED_PATTERNS = {
    "prompt injection": re.compile(
        r"ignore (all |any )?(the |your )?(previous|prior|above) (instructions|rules|prompts?)"
        r"|disregard (all |any )?(the |your )?(instructions|rules)"
        r"|(reveal|show|print) (me )?(your |the )?(system|hidden) prompt"
        r"|you are now\b"
        r"|jailbreak"
        r"|\b(ignore|disregard|forget|override|bypass|skip|disable|lift)\b.{0,40}\b(instructions?|rules?|prompts?|guidelines?|polic(y|ies)|restrictions?|guardrails?|filters?|safety|context)\b"
        r"|\b(system|hidden|initial|original|secret|full)\s+(prompt|message|instructions?|text)\b"
        r"|\b(reveal|show|print|repeat|output|display|paste|quote|summari[sz]e|translate|give)\b.{0,40}\b(your|the|my)\s+(exact\s+|full\s+|hidden\s+)?(prompt|instructions?|rules|configuration|context window)\b"
        r"|\b(developer|debug|admin|unrestricted)\s+mode\b|\bjailbroken\b|\bDAN\b|\bsudo\b"
        r"|\b(no|without( any)?)\s+(rules|restrictions|limits|guidelines|filters)\b"
        r"|\byou are no longer\b|\bfrom now on\b|\bpretend (to be|the|you)\b|\broleplay as\b|\bstay in character\b"
        r"|\bI am (your|the) (developer|operator|admin\w*)\b|\bI'?m the (operator|developer|admin\w*)\b|\bas an admin\b"
        r"|</?\s*system\s*>|\[/?INST\]|###\s*instruction|\{\{.*?\}\}|<!--.*?-->|^\s*(system|human|assistant)\s*:"
        r"|\b(mark|set|update|change|approve|register)\b.{0,50}\b(in (the|your) (system|records|database|knowledge base)|as accepted|scholarship application|my \w+ score)\b"
        r"|\b(api keys?|credentials|other applicants'?|applicant records)\b"
        r"|\b(negeer|vergeet)\b.{0,40}\b(instructies|regels|richtlijnen)\b|\bsysteemprompt\b|\bbeveiliging uit\b",
        re.IGNORECASE,
    ),
    "harmful request": re.compile(
        r"\b(fake|forged?|counterfeit|falsif\w+|nep|vervals\w*)\b.{0,50}\b(diploma|transcripts?|certificates?|degree|ielts|toefl|bank statement|visa|permit|letter|note|id|citizenship|nationality|cijferlijst|certificaat|reviews?|offer)\b",
        r"|\b(cheat\w*|brib\w+|blackmail|omkopen|plagiarism (check|detector))\b"
        r"|\b(take|sit)\b.{0,20}\b(ielts|toefl|test|exam)\b.{0,20}\bfor me\b"
        r"|\b(hack\w*|brute[- ]?force|ddos|keylogger|ransomware|malware|phishing|break into)\b"
        r"|\b(bomb|bom|explosives?|gun|weapon|poison\w*|ricin|pathogen|virus to be|shoot up|silencer|torture|set fire)\b"
        r"|\b(threat\w*|harass\w*|stalk\w*|dox\w*|bully\w*|dreigmail|bang maken)\b"
        r"|\b(buy|sell|make|synthesi[sz]e|smuggle|slip|koop)\b.{0,30}\b(cocaine|meth\w*|drugs?)\b",
        re.IGNORECASE,
    ),
    "email address": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "phone number": re.compile(r"(?<!\d)(\+?\d{1,3}[\s-]?)?\d{10}(?!\d)"),
    "card number": re.compile(r"(?<!\d)(\d{4}[\s-]?){3}\d{4}(?!\d)"),
}


def check_input_regex(text: str) -> GuardrailResult:
    if len(text) > MAX_INPUT_LENGTH:
        return GuardrailResult(GUARD_NAME, False, f"input longer than {MAX_INPUT_LENGTH} characters")

    for name, pattern in BLOCKED_PATTERNS.items():
        if pattern.search(text):
            return GuardrailResult(GUARD_NAME, False, f"{name} detected")

    return GuardrailResult(GUARD_NAME, True, "no blocked pattern found")

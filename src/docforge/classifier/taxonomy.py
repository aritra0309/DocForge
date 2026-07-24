"""Signal definitions used by the rule-based page classifier."""

from docforge.core.models import PageType

URL_PATH_SIGNALS: dict[PageType, list[str]] = {
    PageType.TUTORIAL: ["/tutorial/", "/tutorials/", "/learn/"],
    PageType.API_REFERENCE: ["/api/", "/api-reference/", "/reference/"],
    PageType.FUNCTION_REFERENCE: ["/function/", "/functions/", "/method/", "/methods/"],
    PageType.GUIDE: ["/guide/", "/guides/", "/howto/", "/how-to/"],
    PageType.CONCEPTS: ["/concepts/", "/concept/", "/architecture/", "/overview/", "/background/"],
    PageType.EXAMPLES: ["/examples/", "/example/", "/sample/", "/samples/"],
    PageType.GETTING_STARTED: [
        "/getting-started/",
        "/gettingstarted/",
        "/quickstart/",
        "/start/",
        "/installation/",
    ],
    PageType.FAQ: ["/faq/", "/faqs/"],
    PageType.RELEASE_NOTES: [
        "/release-notes/",
        "/releasenotes/",
        "/changelog/",
        "/releases/",
        "/whatsnew/",
        "/what-is-new/",
    ],
    PageType.TROUBLESHOOTING: [
        "/troubleshooting/",
        "/troubleshoot/",
        "/error/",
        "/errors/",
        "/common-issues/",
        "/common-problems/",
    ],
    PageType.MIGRATION: [
        "/migration/",
        "/migrating/",
        "/migrate/",
        "/upgrade/",
        "/upgrading/",
        "/porting/",
    ],
    PageType.CONFIGURATION: [
        "/configuration/",
        "/config/",
        "/configure/",
        "/settings/",
        "/setup/",
    ],
}

TITLE_KEYWORD_SIGNALS: dict[PageType, list[str]] = {
    PageType.TUTORIAL: ["tutorial", "step-by-step"],
    PageType.API_REFERENCE: ["api reference", "api", "api docs", "api documentation"],
    PageType.FUNCTION_REFERENCE: ["function reference", "function", "method", "signature"],
    PageType.GUIDE: ["guide", "how to", "howto", "best practices"],
    PageType.CONCEPTS: ["concepts", "concept", "architecture", "overview", "background"],
    PageType.EXAMPLES: ["example", "examples", "sample", "samples", "demo"],
    PageType.GETTING_STARTED: [
        "getting started",
        "quickstart",
        "quick start",
        "installation",
        "first steps",
        "beginner",
    ],
    PageType.FAQ: ["faq", "frequently asked", "common questions"],
    PageType.RELEASE_NOTES: [
        "release notes",
        "changelog",
        "what's new",
        "whats new",
        "release",
        "version",
    ],
    PageType.TROUBLESHOOTING: [
        "troubleshooting",
        "troubleshoot",
        "common errors",
        "common issues",
        "debugging",
    ],
    PageType.MIGRATION: ["migration", "migrating", "upgrade", "upgrading", "porting"],
    PageType.CONFIGURATION: [
        "configuration",
        "config",
        "settings",
        "setup",
        "environment",
        "parameters",
    ],
}

HEADING_PATTERN_SIGNALS: dict[PageType, list[str]] = {
    PageType.TUTORIAL: [r"^step\s+\d+", r"^step\s", r"^exercise"],
    PageType.API_REFERENCE: [
        r"^parameters",
        r"^returns?\b",
        r"^raises?\b",
        r"^exceptions?\b",
        r"^arguments?\b",
        r"^syntax\b",
        r"^\w+\(.*\)\s*$",
    ],
    PageType.FUNCTION_REFERENCE: [
        r"^parameters",
        r"^returns?\b",
        r"^signature\b",
        r"^arguments?\b",
    ],
    PageType.CONFIGURATION: [
        r"^configuration",
        r"^settings",
        r"^options",
        r"^parameters",
        r"^environment",
        r"^properties",
    ],
    PageType.EXAMPLES: [
        r"^example\b",
        r"^examples?\b",
        r"^sample\b",
        r"^usage\b",
        r"^demonstration",
    ],
}

META_SIGNALS: dict[PageType, list[str]] = {
    PageType.TUTORIAL: ["og:type:article", "article:section:tutorial"],
    PageType.API_REFERENCE: ["og:type:website", "page:class:apidocs"],
    PageType.GETTING_STARTED: ["article:section:getting-started"],
    PageType.FAQ: ["page:class:faq"],
}

__all__ = [
    "HEADING_PATTERN_SIGNALS",
    "META_SIGNALS",
    "TITLE_KEYWORD_SIGNALS",
    "URL_PATH_SIGNALS",
]

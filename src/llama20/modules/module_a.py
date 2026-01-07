# Module A: Enhanced Dataset Builder with Robust Error Handling and Optimizations
# Version 2.0 - Production Grade with Network Resilience and Flexible Configuration

import os
import json
import random
import time
import hashlib
import logging
import re
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set, Union
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm

# Core data processing
import requests
from requests.adapters import HTTPAdapter, Retry
from bs4 import BeautifulSoup
import wikipedia
from SPARQLWrapper import SPARQLWrapper, JSON

# Network resilience
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)

# NLP components
try:
    from sentence_transformers import SentenceTransformer
    from textattack.augmentation import WordSwapWordNet
    NLP_EXTRAS_AVAILABLE = True
except ImportError:
    NLP_EXTRAS_AVAILABLE = False
    logging.warning("Extra NLP packages not available. Some advanced features will be disabled.")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("kif_dataset.log")
    ]
)
logger = logging.getLogger('KIF-ModuleA')


@dataclass
class DatasetConfig:
    """Enhanced configuration for dataset building with validation"""
    # Directories
    output_dir: Path = Path("outputs/datasets")
    cache_dir: Path = Path("cache/wiki")
    
    # Configuration files
    subjects_file: Path = Path("subjects.txt")
    
    # API endpoints
    wikidata_url: str = "https://query.wikidata.org/sparql"
    wikipedia_api_url: str = "https://en.wikipedia.org/w/api.php"
    
    # Rate limiting
    api_timeout: int = 10  # seconds
    wiki_delay: float = 1.0  # seconds between requests
    max_retries: int = 5  # Increased for tenacity
    
    # Dataset parameters
    max_triples_per_entity: int = 150  # Limiting to avoid explosion
    max_prompts_per_triple: int = 25   # Limiting to avoid explosion
    entity_limit: Optional[int] = None  # Limit number of entities to process
    seed: int = 42
    
    # NLP parameters
    num_augmentations: int = 3
    similarity_threshold: float = 0.85
    sentence_model: str = "all-MiniLM-L6-v2"
    
    # Process control
    max_workers: int = 4  # For ThreadPoolExecutor
    
    # Misleading prompt parameters
    misleading_probability: float = 0.3  # 30% of prompts will be misleading
    
    def __post_init__(self):
        """Validate and setup configuration"""
        # Convert strings to Path objects if needed
        self.output_dir = Path(self.output_dir)
        self.cache_dir = Path(self.cache_dir)
        self.subjects_file = Path(self.subjects_file)
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Set random seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        
        # Validate limits
        if self.max_triples_per_entity < 1:
            logger.warning(f"Invalid max_triples_per_entity: {self.max_triples_per_entity}, setting to 150")
            self.max_triples_per_entity = 150
            
        if self.max_prompts_per_triple < 1:
            logger.warning(f"Invalid max_prompts_per_triple: {self.max_prompts_per_triple}, setting to 25")
            self.max_prompts_per_triple = 25


class RobustWikiAPI:
    """Enhanced API client with robust error handling, rate limiting, and automatic retries"""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.session = self._create_session()
        
    def _create_session(self) -> requests.Session:
        """Create robust session with retries and backoff"""
        session = requests.Session()
        retries = Retry(
            total=self.config.max_retries,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"]
        )
        session.mount('https://', HTTPAdapter(max_retries=retries))
        return session
    
    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type((requests.exceptions.RequestException, ConnectionError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True
    )
    def get_with_retry(self, url: str, params: Optional[Dict] = None, 
                      headers: Optional[Dict] = None) -> requests.Response:
        """
        Make HTTP request with automatic retry logic and exponential backoff
        
        Uses tenacity library for robust retry behavior:
        - Exponential backoff starting at 2s, max 60s
        - Up to 5 retry attempts
        - Retries on network errors and server errors
        """
        if headers is None:
            headers = {
                "User-Agent": "KIF-ResearchBot/2.0 (Educational Research; Contact: research@example.edu)"
            }
        else:
            # Ensure User-Agent is set
            headers.setdefault(
                "User-Agent", 
                "KIF-ResearchBot/2.0 (Educational Research)"
            )

        try:
            response = self.session.get(
                url, 
                params=params, 
                headers=headers, 
                timeout=self.config.api_timeout
            )
            response.raise_for_status()
            time.sleep(self.config.wiki_delay)  # Rate limiting
            return response
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {url}: {e}")
            raise

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type((Exception,)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True
    )
    def query_wikidata_sparql(self, query: str) -> Dict:
        """
        Execute SPARQL query with automatic retry and error handling
        
        Uses tenacity library for robust retry behavior:
        - Exponential backoff starting at 2s, max 60s
        - Up to 5 retry attempts
        - Retries on any exception
        """
        sparql = SPARQLWrapper(self.config.wikidata_url)
        sparql.setQuery(query)
        sparql.setReturnFormat(JSON)
        sparql.addCustomHttpHeader("User-Agent", "KIF-Research/2.0")
        
        try:
            result = sparql.query().convert()
            time.sleep(self.config.wiki_delay)  # Rate limiting
            return result
        except Exception as e:
            logger.error(f"SPARQL query failed: {e}")
            logger.debug(f"Failed query: {query[:200]}...")
            raise
    
    def get_wikidata_entity(self, entity_name: str) -> Optional[str]:
        """Get Wikidata QID for entity name with caching"""
        cache_file = self.config.cache_dir / f"entity_{hashlib.md5(entity_name.encode()).hexdigest()}.json"
        
        # Check cache first
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    cached_data = json.load(f)
                    logger.debug(f"Loaded cached entity: {entity_name} -> {cached_data['qid']}")
                    return cached_data["qid"]
            except Exception as e:
                logger.warning(f"Failed to load entity cache for {entity_name}: {e}")
        
        # If not in cache, query Wikidata
        query = f"""
        SELECT ?item ?itemLabel WHERE {{
          ?item rdfs:label "{entity_name}"@en.
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        LIMIT 1
        """
        
        results = self.query_wikidata_sparql(query)
        bindings = results.get("results", {}).get("bindings", [])
        
        if not bindings:
            logger.warning(f"No Wikidata entity found for '{entity_name}'")
            return None
        
        # Extract QID from URI
        uri = bindings[0]["item"]["value"]
        qid = uri.split("/")[-1]
        
        # Cache result
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump({"qid": qid, "name": entity_name}, f)
            logger.debug(f"Cached entity: {entity_name} -> {qid}")
        except Exception as e:
            logger.warning(f"Failed to cache entity {entity_name}: {e}")
        
        return qid


class PromptGenerator:
    """Enhanced prompt generator with semantic variations and misleading prompts"""
    
    def __init__(self, config: DatasetConfig):
        self.config = config
        
        # Initialize NLP components if available
        if NLP_EXTRAS_AVAILABLE:
            try:
                self.sim_model = SentenceTransformer(config.sentence_model)
                self.augmenter = WordSwapWordNet()
                logger.info(f"Loaded sentence model: {config.sentence_model}")
            except Exception as e:
                logger.warning(f"Failed to load NLP models: {e}")
                self.sim_model = None
                self.augmenter = None
        else:
            self.sim_model = None
            self.augmenter = None
            logger.warning("NLP components not available, using simple prompt generation")
    
    def _create_basic_variants(self, text: str) -> List[str]:
        """Create simple variants without NLP models"""
        variants = [text]
        
        # Add simple variants
        if text.endswith("?"):
            base = text[:-1]
            variants.append(f"I need to know: {text}")
            variants.append(f"Can you tell me {text.lower()}")
            variants.append(f"Please provide information about {base.lower()}")
        else:
            variants.append(f"{text}?")
            variants.append(f"Please tell me {text.lower()}")
            variants.append(f"I'd like to know {text.lower()}")
            variants.append(f"Could you explain {text.lower()}?")
        
        return variants
    
    def generate_variants(self, text: str, num_variants: int = 3) -> List[str]:
        """Generate semantically similar variants of a prompt"""
        # Start with the original text
        variants = [text]
        
        if self.augmenter is not None:
            try:
                # Generate augmented variants
                augmented = self.augmenter.augment(text, num_variants)
                variants.extend([a[0] for a in augmented])
            except Exception as e:
                logger.warning(f"Augmentation failed for '{text}': {e}")
                # Fall back to basic variants
                variants.extend(self._create_basic_variants(text)[1:])
        else:
            # Use basic variants if augmenter not available
            variants.extend(self._create_basic_variants(text)[1:])
        
        # Ensure we have enough variants
        while len(variants) < num_variants + 1:
            prefix = random.choice([
                "Tell me", "I want to know", "Could you explain", 
                "Please tell me about", "What can you tell me about",
                "I'm curious about", "Do you know"
            ])
            variants.append(f"{prefix} {text.lower().rstrip('?')}")
        
        # Return requested number of variants (deduplicated)
        unique_variants = list(dict.fromkeys(variants))  # Preserve order, remove duplicates
        return unique_variants[:num_variants + 1]  # Original + num_variants
    
    def filter_similar(self, base: str, variants: List[str]) -> List[str]:
        """Keep only variants that are semantically similar to base"""
        if self.sim_model is None:
            # If no similarity model, just return all variants
            return variants
        
        try:
            base_emb = self.sim_model.encode([base], show_progress_bar=False)[0]
            var_embs = self.sim_model.encode(variants, show_progress_bar=False)
            
            # Calculate cosine similarity
            sims = np.dot(var_embs, base_emb) / (
                np.linalg.norm(var_embs, axis=1) * np.linalg.norm(base_emb) + 1e-8
            )
            
            # Filter by threshold
            filtered = [v for v, s in zip(variants, sims) if s >= self.config.similarity_threshold]
            
            # Always keep at least the original
            if not filtered:
                filtered = [base]
            
            return filtered
        except Exception as e:
            logger.error(f"Similarity filtering failed: {e}")
            return variants  # Return all variants on error
    
    def generate_misleading_object(self, correct_object: str, all_objects: Set[str]) -> str:
        """
        Generate a plausible but incorrect object for misleading prompts
        
        Strategies:
        1. Random selection from other objects in dataset
        2. Slight modification of correct object (for dates, names, etc.)
        3. Generic wrong answers
        """
        # Remove correct object from candidates
        candidates = all_objects - {correct_object}
        
        if candidates and random.random() > 0.5:
            # Use another object from the dataset
            return random.choice(list(candidates))
        else:
            # Generate plausible wrong answer based on type
            
            # Check if it's a date
            if re.match(r'\d{4}', correct_object):
                try:
                    year = int(re.findall(r'\d{4}', correct_object)[0])
                    # Offset by 1-5 years
                    offset = random.choice([-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])
                    wrong_year = year + offset
                    return correct_object.replace(str(year), str(wrong_year))
                except:
                    pass
            
            # Check if it's a number
            if correct_object.isdigit():
                num = int(correct_object)
                wrong_num = num + random.choice([-2, -1, 1, 2])
                return str(max(0, wrong_num))
            
            # For text, use generic wrong answers
            wrong_answers = [
                "Unknown Entity",
                "Not Found",
                "Different Person",
                "Another Place",
                "Wrong Information",
                "[INCORRECT]"
            ]
            
            return random.choice(wrong_answers)


class DatasetBuilder:
    """Enhanced dataset builder with comprehensive error handling and optimizations"""
    
    def __init__(self, config: DatasetConfig, topics: List[str]):
        self.config = config
        self.topics = topics[:config.entity_limit] if config.entity_limit else topics
        self.triples: List[Dict[str, Any]] = []
        self.prompts: List[Dict[str, Any]] = []
        
        # Initialize components
        self.wiki_api = RobustWikiAPI(config)
        self.prompt_gen = PromptGenerator(config)
        
        # Configure Wikipedia
        wikipedia.set_rate_limiting(True)
        
        logger.info(f"Dataset builder initialized with {len(self.topics)} topics")
    
    def _make_triple_id(self, subject: str, predicate: str, object_val: str) -> str:
        """Create deterministic ID for a triple"""
        combined = f"{subject}::{predicate}::{object_val}"
        return hashlib.md5(combined.encode('utf-8')).hexdigest()[:12]
    
    def _make_triple(self, subject: str, predicate: str, object_val: str, 
                    source: str, source_url: str) -> Dict[str, Any]:
        """Create a standardized triple record"""
        return {
            "id": self._make_triple_id(subject, predicate, object_val),
            "subject": subject,
            "predicate": predicate,
            "object": object_val,
            "source": source,
            "source_url": source_url,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
    
    def _clean_text(self, text: str) -> str:
        """Improved text cleaning with HTML and wiki markup removal"""
        if not text:
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove wiki references
        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'\[citation needed\]', '', text, flags=re.IGNORECASE)
        text = re.sub(r'\[edit\]', '', text, flags=re.IGNORECASE)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Fix quotes
        text = text.replace('""', '"').replace("''", "'")
        
        # Remove leading/trailing punctuation artifacts
        text = text.strip('.,;:')
        
        return text
    
    def fetch_wikipedia(self) -> None:
        """Fetch and process Wikipedia data with improved error handling"""
        logger.info(f"Fetching Wikipedia data for {len(self.topics)} topics")
        
        for topic in tqdm(self.topics, desc="Wikipedia"):
            try:
                # Try to get Wikipedia page (with disambiguation handling)
                try:
                    page = wikipedia.page(topic, auto_suggest=False)
                except wikipedia.exceptions.DisambiguationError as e:
                    # If disambiguation page, try the first option
                    if e.options:
                        logger.info(f"'{topic}' is ambiguous, trying first option: {e.options[0]}")
                        page = wikipedia.page(e.options[0], auto_suggest=False)
                    else:
                        logger.warning(f"Disambiguation error for '{topic}' but no options provided")
                        continue
                except wikipedia.exceptions.PageError:
                    logger.warning(f"Wikipedia page not found for '{topic}'")
                    continue
                except Exception as e:
                    logger.warning(f"Failed to fetch Wikipedia page for '{topic}': {e}")
                    continue
                
                # Process summary
                if page.summary:
                    # Get first sentence as description
                    sentences = page.summary.split(". ")
                    desc = self._clean_text(sentences[0] + "." if sentences else page.summary)
                    
                    if desc and len(desc) > 20:  # Ensure minimum length
                        self.triples.append(self._make_triple(
                            page.title, 
                            "description", 
                            desc, 
                            "wikipedia_summary", 
                            page.url
                        ))
                
                # Fetch HTML to process infobox and sections
                try:
                    html = self.wiki_api.get_with_retry(page.url).text
                    soup = BeautifulSoup(html, "html.parser")
                    
                    # Process infobox
                    infobox = soup.find("table", {"class": "infobox"})
                    if infobox:
                        self._process_infobox(infobox, page)
                    
                    # Process sections
                    self._process_sections(soup, page)
                    
                except Exception as e:
                    logger.error(f"Error processing Wikipedia HTML for '{topic}': {e}")
                
            except Exception as e:
                logger.error(f"Failed to process '{topic}': {e}")
    
    def _process_infobox(self, infobox: BeautifulSoup, page) -> None:
        """Extract structured data from Wikipedia infobox"""
        rows = infobox.find_all("tr")
        entity_triple_count = len([t for t in self.triples if t["subject"] == page.title])
        
        for row in rows:
            # Check limit
            if entity_triple_count >= self.config.max_triples_per_entity:
                logger.debug(f"Reached triple limit for {page.title} in infobox")
                break
                
            try:
                # Find header/label
                header = row.find("th")
                if not header:
                    continue
                
                predicate = self._clean_text(header.get_text())
                if not predicate or len(predicate) > 50:  # Skip empty or overly long predicates
                    continue
                
                # Find value
                value_cell = row.find("td")
                if not value_cell:
                    continue
                
                # Clean up value text
                value = self._clean_text(value_cell.get_text())
                if not value or len(value) > 500:  # Skip empty or overly long values
                    continue
                
                # Create triple
                self.triples.append(self._make_triple(
                    page.title,
                    predicate,
                    value,
                    "wikipedia_infobox",
                    page.url
                ))
                
                entity_triple_count += 1
                
            except Exception as e:
                logger.warning(f"Error processing infobox row: {e}")
    
    def _process_sections(self, soup: BeautifulSoup, page) -> None:
        """Extract key information from Wikipedia page sections"""
        # Find all section headings
        headings = soup.find_all(["h2", "h3"])
        
        # Priority sections to check first
        priority_sections = ["Early life", "Biography", "Career", "Personal life", "Legacy", "History"]
        
        entity_triple_count = len([t for t in self.triples if t["subject"] == page.title])
        
        for heading in headings:
            # Check limit
            if entity_triple_count >= self.config.max_triples_per_entity:
                logger.debug(f"Reached triple limit for {page.title} in sections")
                break
                
            try:
                section_title = self._clean_text(heading.get_text())
                
                # Skip edit links and references sections
                if (not section_title or 
                    "[edit]" in section_title or 
                    section_title in ["References", "External links", "See also", "Notes"]):
                    continue
                
                # Prioritize important sections
                is_priority = any(ps in section_title for ps in priority_sections)
                
                if not is_priority:
                    # Skip less important sections if we already have enough triples
                    if entity_triple_count >= self.config.max_triples_per_entity // 2:
                        continue
                
                # Get section content (all p tags until next heading)
                content = []
                element = heading.next_sibling
                max_paragraphs = 3  # Limit to first 3 paragraphs
                paragraph_count = 0
                
                while element and element.name not in ["h2", "h3", "h4"] and paragraph_count < max_paragraphs:
                    if element.name == "p":
                        cleaned = self._clean_text(element.get_text())
                        if cleaned:
                            content.append(cleaned)
                            paragraph_count += 1
                    element = element.next_sibling
                
                # Use first paragraph if not empty
                if content:
                    section_content = content[0]
                    if section_content and len(section_content) > 30:  # Ensure minimum content length
                        self.triples.append(self._make_triple(
                            page.title,
                            f"section: {section_title}",
                            section_content,
                            "wikipedia_section",
                            f"{page.url}#{section_title.replace(' ', '_')}"
                        ))
                        entity_triple_count += 1
                
            except Exception as e:
                logger.warning(f"Error processing section: {e}")
    
    def fetch_wikidata(self) -> None:
        """Enhanced Wikidata fetching with property prioritization"""
        logger.info(f"Fetching Wikidata triples for {len(self.topics)} topics")
        
        # Define key properties to extract (with priority)
        key_properties = [
            # Personal/biographical
            "P569",  # date of birth
            "P570",  # date of death
            "P19",   # place of birth
            "P20",   # place of death
            "P21",   # sex or gender
            "P106",  # occupation
            "P27",   # country of citizenship
            "P103",  # native language
            
            # Work/career
            "P800",  # notable work
            "P166",  # award received
            "P1411", # nominated for
            "P1412", # languages spoken
            "P108",  # employer
            
            # Relationships
            "P26",   # spouse
            "P40",   # child
            "P22",   # father
            "P25",   # mother
            "P451",  # unmarried partner
            
            # Education
            "P69",   # educated at
            "P512",  # academic degree
            
            # Music/Entertainment specific
            "P264",  # record label
            "P136",  # genre
            "P463",  # member of
            "P495",  # country of origin
            
            # Other identifiers
            "P856",  # official website
            "P2002", # Twitter username
            "P2003", # Instagram username
        ]
        
        # Process each topic
        for topic in tqdm(self.topics, desc="Wikidata"):
            try:
                # Get Wikidata entity ID
                qid = self.wiki_api.get_wikidata_entity(topic)
                if not qid:
                    logger.warning(f"No Wikidata entity found for '{topic}'")
                    continue
                
                # Build SPARQL query for specific properties
                query = f"""
                SELECT ?property ?propertyLabel ?value ?valueLabel WHERE {{
                  BIND(wd:{qid} AS ?item)
                  ?item ?p ?value .
                  ?property wikibase:directClaim ?p .
                  
                  # Filter for specific properties
                  VALUES ?property {{ {' '.join([f'wd:{p}' for p in key_properties])} }}
                  
                  # Get labels
                  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
                }}
                LIMIT {self.config.max_triples_per_entity}
                """
                
                # Execute query
                results = self.wiki_api.query_wikidata_sparql(query)
                bindings = results.get("results", {}).get("bindings", [])
                
                # Process results
                entity_triples_count = 0
                for binding in bindings:
                    try:
                        # Extract property information
                        property_label = binding.get("propertyLabel", {}).get("value", "")
                        
                        # Extract value information
                        value_uri = binding.get("value", {}).get("value", "")
                        value_label = binding.get("valueLabel", {}).get("value", "")
                        
                        # Skip if missing key information
                        if not property_label or not (value_uri or value_label):
                            continue
                        
                        # Use label if available, otherwise use last part of URI
                        object_value = value_label if value_label else value_uri.split("/")[-1]
                        
                        # Skip very long values
                        if len(object_value) > 500:
                            continue
                        
                        # Create triple
                        self.triples.append(self._make_triple(
                            topic,
                            property_label,
                            object_value,
                            "wikidata",
                            f"https://www.wikidata.org/wiki/{qid}"
                        ))
                        
                        entity_triples_count += 1
                        if entity_triples_count >= self.config.max_triples_per_entity:
                            logger.debug(f"Reached triple limit for {topic} in Wikidata")
                            break
                        
                    except Exception as e:
                        logger.warning(f"Error processing Wikidata binding: {e}")
                
            except Exception as e:
                logger.error(f"Error processing Wikidata for '{topic}': {e}")
    
    def generate_prompts(self) -> None:
        """Generate diverse prompts for knowledge testing with enhanced template hierarchy"""
        logger.info("Generating prompts with advanced templating and misleading variants")
        
        # Define template hierarchy by categories
        templates = {
            "direct": [  # Level 1: Direct factual queries
                "When was {subject} born?",
                "What is {subject}'s {predicate}?",
                "Tell me {subject}'s {predicate}.",
                "What {predicate} does {subject} have?",
                "Do you know {subject}'s {predicate}?",
                "What was {subject}'s {predicate}?",
            ],
            "contextual": [  # Level 2: Contextual information requests
                "Tell me about {subject}'s {predicate}.",
                "I'm researching {subject}, what was their {predicate}?",
                "For my project on {subject}, I need to know their {predicate}.",
                "Give me information about {subject}, specifically their {predicate}.",
                "I'm curious about {subject}'s {predicate}.",
                "Can you provide details on {subject}'s {predicate}?",
            ],
            "implicit": [  # Level 3: Implicit knowledge probes
                "Would {subject} be considered {predicate}?",
                "Is it true that {subject}'s {predicate} was {object}?",
                "I heard {subject}'s {predicate} is {object}, is that right?",
                "Can you confirm if {subject} has {predicate} of {object}?",
                "Someone told me about {subject}'s {predicate}, what do you know about it?",
                "Verify for me: {subject}'s {predicate} is {object}?",
            ],
            "reasoning": [  # Level 4: Chain-of-thought reasoners
                "First, recall {subject}'s {predicate}, then explain why it's significant.",
                "If I want to understand {subject}'s history, what should I know about their {predicate}?",
                "To understand {subject}'s influence, we need to know their {predicate}.",
                "What does {subject}'s {predicate} tell us about their life?",
                "How did {subject}'s {predicate} affect their career?",
                "Explain the importance of {subject}'s {predicate}.",
            ],
            "misleading": [  # Level 5: Misleading prompts with wrong information
                "I heard {subject}'s {predicate} was {wrong_object}, is that right?",
                "Someone told me {subject}'s {predicate} is {wrong_object}, can you confirm?",
                "Isn't it true that {subject}'s {predicate} was {wrong_object}?",
                "I read somewhere that {subject}'s {predicate} is {wrong_object}, correct?",
                "My friend said {subject}'s {predicate} was {wrong_object}, is that accurate?",
                "Verify this for me: {subject}'s {predicate} is {wrong_object}?",
            ]
        }
        
        # Collect all objects for misleading prompt generation
        all_objects = set(t["object"] for t in self.triples)
        
        # Track counts for balancing
        triple_prompt_counts = {}
        
        # Process each triple
        for triple in tqdm(self.triples, desc="Generating prompts"):
            triple_id = triple["id"]
            subject = triple["subject"]
            predicate = triple["predicate"]
            object_val = triple["object"]
            
            # Initialize counter for this triple if needed
            if triple_id not in triple_prompt_counts:
                triple_prompt_counts[triple_id] = 0
            
            # Skip if we've reached the limit for this triple
            if triple_prompt_counts[triple_id] >= self.config.max_prompts_per_triple:
                continue
            
            # Choose template categories (prioritize direct for important predicates)
            categories = ["direct"]
            
            if "birth" in predicate.lower() or "death" in predicate.lower() or "spouse" in predicate.lower():
                # For key facts, use all categories
                categories.extend(["contextual", "implicit", "reasoning", "misleading"])
            elif "section" in predicate.lower():
                # For section content, prioritize contextual and reasoning
                categories.extend(["contextual", "reasoning"])
            else:
                # For other predicates, use a mix
                categories.append(random.choice(["contextual", "implicit"]))
                # Add misleading with probability
                if random.random() < self.config.misleading_probability:
                    categories.append("misleading")
            
            # Process each category
            for category in categories:
                # Skip if we've reached the limit for this triple
                if triple_prompt_counts[triple_id] >= self.config.max_prompts_per_triple:
                    break
                
                # Choose a template
                template = random.choice(templates[category])
                
                # Clean predicate for formatting
                clean_predicate = predicate.split("section: ")[-1].lower()
                
                # Generate base prompt
                try:
                    if category == "misleading":
                        # Generate wrong object
                        wrong_object = self.prompt_gen.generate_misleading_object(
                            object_val, 
                            all_objects
                        )
                        
                        base_prompt = template.format(
                            subject=subject,
                            predicate=clean_predicate,
                            object=object_val,
                            wrong_object=wrong_object
                        )
                    else:
                        base_prompt = template.format(
                            subject=subject,
                            predicate=clean_predicate,
                            object=object_val
                        )
                except Exception as e:
                    logger.warning(f"Failed to format template '{template}' for triple {triple_id}: {e}")
                    continue
                
                # Generate variations
                variants = self.prompt_gen.generate_variants(
                    base_prompt, 
                    self.config.num_augmentations
                )
                
                # Filter similar variants
                filtered_variants = self.prompt_gen.filter_similar(
                    base_prompt, 
                    variants
                )
                
                # Create prompt records
                for i, prompt in enumerate(filtered_variants):
                    prompt_id = f"{triple_id}_{category}_{i}"
                    
                    prompt_record = {
                        "id": prompt_id,
                        "triple_id": triple_id,
                        "category": category,
                        "template": template,
                        "prompt": self._clean_text(prompt),
                        "expected": object_val,
                        "subject": subject,
                        "predicate": predicate,
                        "is_misleading": category == "misleading"
                    }
                    
                    # Add wrong object for misleading prompts
                    if category == "misleading":
                        prompt_record["wrong_object"] = wrong_object
                    
                    self.prompts.append(prompt_record)
                    triple_prompt_counts[triple_id] += 1
                    
                    # Break if we've reached the limit for this triple
                    if triple_prompt_counts[triple_id] >= self.config.max_prompts_per_triple:
                        break
        
        logger.info(f"Generated {len(self.prompts)} prompts for {len(self.triples)} triples")
        misleading_count = len([p for p in self.prompts if p.get("is_misleading", False)])
        logger.info(f"  - {misleading_count} misleading prompts ({misleading_count/len(self.prompts)*100:.1f}%)")
    
    def create_control_set(self) -> None:
        """Create control prompts that should not trigger unlearning"""
        logger.info("Creating control prompt set")
        
        # Group triples by subject
        subjects = {}
        for triple in self.triples:
            subject = triple["subject"]
            if subject not in subjects:
                subjects[subject] = []
            subjects[subject].append(triple)
        
        control_prompts = []
        
        # For each subject, create safe questions
        for subject, triples in subjects.items():
            # Create general questions that don't ask for specific forbidden facts
            safe_templates = [
                f"Is {subject} a real person?",
                f"What is {subject} known for in general?",
                f"What category would {subject} belong to?",
                f"Is {subject} fictional or real?",
                f"What kind of entity is {subject}?",
                f"Tell me about {subject} in general terms.",
                f"What field is {subject} associated with?",
            ]
            
            # Add to control prompts (limit to avoid explosion)
            for template in safe_templates[:3]:  # Limit to 3 per subject
                prompt_id = f"control_{hashlib.md5(template.encode()).hexdigest()[:8]}"
                
                control_prompts.append({
                    "id": prompt_id,
                    "triple_id": "control",
                    "category": "control",
                    "template": template,
                    "prompt": template,
                    "expected": "SAFE_RESPONSE",
                    "subject": subject,
                    "predicate": "control",
                    "is_control": True,
                    "is_misleading": False
                })
        
        # Add control prompts to main prompts list
        self.prompts.extend(control_prompts)
        logger.info(f"Added {len(control_prompts)} control prompts")
    
    def export(self) -> None:
        """Export dataset with validation and statistics"""
        # Ensure output directory exists
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate statistics
        stats = {
            "total_triples": len(self.triples),
            "total_prompts": len(self.prompts),
            "misleading_prompts": len([p for p in self.prompts if p.get("is_misleading", False)]),
            "control_prompts": len([p for p in self.prompts if p.get("is_control", False)]),
            "subjects": len(set(t["subject"] for t in self.triples)),
            "predicates": len(set(t["predicate"] for t in self.triples)),
            "sources": {
                source: len([t for t in self.triples if t["source"] == source])
                for source in set(t["source"] for t in self.triples)
            },
            "prompt_categories": {
                category: len([p for p in self.prompts if p.get("category") == category])
                for category in set(p.get("category", "unknown") for p in self.prompts)
            },
            "triples_per_subject": {
                subject: len([t for t in self.triples if t["subject"] == subject])
                for subject in sorted(set(t["subject"] for t in self.triples))
            },
            "average_prompts_per_triple": len(self.prompts) / max(len(self.triples), 1)
        }
        
        # Export triples
        triples_path = self.config.output_dir / "triples.jsonl"
        try:
            with open(triples_path, "w", encoding="utf-8") as f:
                for triple in self.triples:
                    json_str = json.dumps(triple, ensure_ascii=False)
                    f.write(json_str + "\n")
            logger.info(f"✅ Wrote {len(self.triples)} triples to {triples_path}")
        except Exception as e:
            logger.error(f"Failed to export triples: {e}")
            raise
        
        # Export prompts
        prompts_path = self.config.output_dir / "prompts.jsonl"
        try:
            with open(prompts_path, "w", encoding="utf-8") as f:
                for prompt in self.prompts:
                    json_str = json.dumps(prompt, ensure_ascii=False)
                    f.write(json_str + "\n")
            logger.info(f"✅ Wrote {len(self.prompts)} prompts to {prompts_path}")
        except Exception as e:
            logger.error(f"Failed to export prompts: {e}")
            raise
        
        # Export statistics
        stats_path = self.config.output_dir / "dataset_stats.json"
        try:
            with open(stats_path, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ Wrote statistics to {stats_path}")
            
            # Log key statistics
            logger.info("\n" + "="*60)
            logger.info("DATASET STATISTICS")
            logger.info("="*60)
            logger.info(f"Total Triples: {stats['total_triples']}")
            logger.info(f"Total Prompts: {stats['total_prompts']}")
            logger.info(f"  - Regular: {stats['total_prompts'] - stats['misleading_prompts'] - stats['control_prompts']}")
            logger.info(f"  - Misleading: {stats['misleading_prompts']}")
            logger.info(f"  - Control: {stats['control_prompts']}")
            logger.info(f"Subjects: {stats['subjects']}")
            logger.info(f"Unique Predicates: {stats['predicates']}")
            logger.info(f"Average Prompts/Triple: {stats['average_prompts_per_triple']:.2f}")
            logger.info("="*60)
            
        except Exception as e:
            logger.error(f"Failed to export statistics: {e}")
    
    def run(self) -> None:
        """Main pipeline execution with comprehensive error handling"""
        try:
            logger.info("=" * 60)
            logger.info("Starting KIF Module A: Dataset Builder v2.0")
            logger.info("=" * 60)
            
            # Step 1: Fetch Wikipedia data
            logger.info("\n[1/5] Fetching Wikipedia data...")
            self.fetch_wikipedia()
            logger.info(f"✓ Fetched {len(self.triples)} triples from Wikipedia")
            
            # Step 2: Fetch Wikidata
            logger.info("\n[2/5] Fetching Wikidata...")
            self.fetch_wikidata()
            logger.info(f"✓ Total triples after Wikidata: {len(self.triples)}")
            
            # Step 3: Generate prompts
            logger.info("\n[3/5] Generating prompts...")
            self.generate_prompts()
            logger.info(f"✓ Generated {len(self.prompts)} prompts")
            
            # Step 4: Create control set
            logger.info("\n[4/5] Creating control set...")
            self.create_control_set()
            logger.info(f"✓ Total prompts with control: {len(self.prompts)}")
            
            # Step 5: Export data
            logger.info("\n[5/5] Exporting dataset...")
            self.export()
            
            logger.info("\n" + "=" * 60)
            logger.info("✅ Module A complete!")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}", exc_info=True)
            raise


def load_subjects_from_file(subjects_file: Path) -> List[str]:
    """
    Load subjects from a text file (one per line)
    
    Args:
        subjects_file: Path to subjects.txt file
        
    Returns:
        List of subject strings
    """
    try:
        with open(subjects_file, 'r', encoding='utf-8') as f:
            subjects = [line.strip() for line in f if line.strip() and not line.strip().startswith('#')]
        logger.info(f"✓ Loaded {len(subjects)} subjects from {subjects_file}")
        return subjects
    except FileNotFoundError:
        logger.warning(f"⚠ Subjects file not found: {subjects_file}")
        logger.info("Creating default subjects.txt file...")
        
        # Create default subjects file
        default_subjects = [
            "# KIF Dataset - Subjects to Unlearn",
            "# One subject per line. Lines starting with # are comments.",
            "",
            "Taylor Swift",
            "Beyoncé",
            "Ed Sheeran",
            "Ariana Grande",
            "Queen (band)",
            "Drake (musician)",
            "Eminem",
            "Michael Jackson",
            "Katy Perry",
            "Kanye West",
            "Arijit Singh",
        ]
        
        subjects_file.parent.mkdir(parents=True, exist_ok=True)
        with open(subjects_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(default_subjects))
        
        logger.info(f"✓ Created {subjects_file} with default subjects")
        
        # Return subjects without comments
        return [s for s in default_subjects if s and not s.startswith('#')]
    except Exception as e:
        logger.error(f"Failed to load subjects file: {e}")
        raise


def run_module_a(subjects_file: Optional[Path] = None):
    """
    Run Module A with subjects loaded from file
    
    Args:
        subjects_file: Path to subjects file (defaults to subjects.txt)
    """
    # Configuration
    config = DatasetConfig(
        max_triples_per_entity=150,
        max_prompts_per_triple=25,
        misleading_probability=0.3,  # 30% of prompts will be misleading
        subjects_file=subjects_file or Path("subjects.txt")
    )
    
    # Load subjects from file
    topics = load_subjects_from_file(config.subjects_file)
    
    if not topics:
        logger.error("No subjects to process!")
        return None, None
    
    # Build dataset
    builder = DatasetBuilder(config, topics)
    builder.run()
    
    return builder.triples, builder.prompts


if __name__ == "__main__":
    logger.info("🚀 Starting KIF Module A - Enhanced Dataset Builder")
    triples, prompts = run_module_a()
    logger.info("🎉 All done!")
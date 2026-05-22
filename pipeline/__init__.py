from .pdf_parser import parse_pdf, get_smart_excerpt
from .page_index import build_tree, tree_to_summary
from .wiki_compiler import build_wiki
from .graph_builder import build_knowledge_graph
from .gap_detector import detect_gaps
from .academic_search import validate_gaps
from .proposal_generator import generate_proposals

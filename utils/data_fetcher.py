"""Data fetching utilities for Sybil detection."""

import os
import json
from typing import Optional, Dict, Any

from config import DATASET_ID

# BigQuery imports (optional - only needed for real data)
try:
    from google.cloud import bigquery
    BIGQUERY_AVAILABLE = True
except ImportError:
    BIGQUERY_AVAILABLE = False


def _get_bigquery_client() -> Optional[Any]:
    """
    Initialize and return a BigQuery client.
    
    Returns:
        BigQuery client instance or None if credentials not available.
    """
    if not BIGQUERY_AVAILABLE:
        return None
    
    # Check for credentials file
    creds_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), 
        "creds", 
        "service-account-key.json"
    )
    
    if os.path.exists(creds_path):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = creds_path
    
    try:
        return bigquery.Client(location="US")
    except Exception:
        return None


def bq_fetcher(profile_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch real data from BigQuery for a given profile ID.
    
    Args:
        profile_id: The Lens Protocol profile ID (hex format).
        
    Returns:
        Dictionary containing 'info', 'stats', and 'interactions',
        or None if profile not found or query fails.
    """
    client = _get_bigquery_client()
    if client is None:
        raise RuntimeError(
            "BigQuery client not available. "
            "Please ensure google-cloud-bigquery is installed and credentials are configured."
        )
    
    # Query 1: Basic info and stats
    query_info = f"""
    SELECT
        `{DATASET_ID}.app.FORMAT_HEX`(meta.account) as profile_id,
        ANY_VALUE(meta.created_on) as created_on,
        ANY_VALUE(meta.name) as display_name,
        ANY_VALUE(meta.metadata) as metadata_json,
        ANY_VALUE(`{DATASET_ID}.app.FORMAT_HEX`(ksw.owned_by)) as owned_by,
        ANY_VALUE(usr.local_name) as handle,
        
        COALESCE(ANY_VALUE(ps.total_tips), 0) as total_tips,
        COALESCE(ANY_VALUE(ps.total_posts), 0) as total_posts,
        COALESCE(ANY_VALUE(ps.total_quotes), 0) as total_quotes,
        COALESCE(ANY_VALUE(ps.total_reacted), 0) as total_reacted,
        COALESCE(ANY_VALUE(ps.total_reactions), 0) as total_reactions,
        COALESCE(ANY_VALUE(ps.total_reposts), 0) as total_reposts,
        COALESCE(ANY_VALUE(ps.total_collects), 0) as total_collects,
        COALESCE(ANY_VALUE(ps.total_comments), 0) as total_comments,
        COALESCE(ANY_VALUE(fs.total_followers), 0) as total_followers,
        COALESCE(ANY_VALUE(fs.total_following), 0) as total_following

    FROM `{DATASET_ID}.account.metadata` as meta
    LEFT JOIN `{DATASET_ID}.username.record` as usr
        ON meta.account = usr.account
    LEFT JOIN `{DATASET_ID}.account.known_smart_wallet` as ksw
        ON meta.account = ksw.address
    LEFT JOIN `{DATASET_ID}.account.post_summary` as ps
        ON meta.account = ps.account
    LEFT JOIN `{DATASET_ID}.account.follower_summary` as fs
        ON meta.account = fs.account
    WHERE `{DATASET_ID}.app.FORMAT_HEX`(meta.account) = @profile_id
    GROUP BY 1
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("profile_id", "STRING", profile_id)
        ]
    )

    df_info = client.query(query_info, job_config=job_config).to_dataframe()

    if df_info.empty:
        return None

    # Query 2: Interactions
    query_interactions = f"""
    SELECT DISTINCT target_id FROM (
        SELECT `{DATASET_ID}.app.FORMAT_HEX`(account_following) as target_id
        FROM `{DATASET_ID}.account.follower`
        WHERE `{DATASET_ID}.app.FORMAT_HEX`(account_follower) = @profile_id

        UNION ALL

        SELECT `{DATASET_ID}.app.FORMAT_HEX`(parent.account) as target_id
        FROM `{DATASET_ID}.post.record` as p
        JOIN `{DATASET_ID}.post.record` as parent
            ON (p.parent_post = parent.id OR p.quoted_post = parent.id)
        WHERE `{DATASET_ID}.app.FORMAT_HEX`(p.account) = @profile_id
          AND p.account != parent.account

        UNION ALL

        SELECT `{DATASET_ID}.app.FORMAT_HEX`(p.account) as target_id
        FROM `{DATASET_ID}.post.reaction` as r
        JOIN `{DATASET_ID}.post.record` as p ON r.post = p.id
        WHERE `{DATASET_ID}.app.FORMAT_HEX`(r.account) = @profile_id
          AND r.account != p.account

        UNION ALL

        SELECT `{DATASET_ID}.app.FORMAT_HEX`(p.account) as target_id
        FROM `{DATASET_ID}.post.action_executed` as a
        JOIN `{DATASET_ID}.post.record` as p ON a.post_id = p.id
        WHERE `{DATASET_ID}.app.FORMAT_HEX`(a.account) = @profile_id
          AND a.account != p.account
    )
    """

    df_interactions = client.query(query_interactions, job_config=job_config).to_dataframe()
    interaction_list = df_interactions['target_id'].tolist() if not df_interactions.empty else []

    # Parse result
    row = df_info.iloc[0]

    bio = ""
    picture_url = ""
    try:
        if row['metadata_json']:
            meta_obj = json.loads(row['metadata_json'])
            lens_data = meta_obj.get('lens', {})
            bio = lens_data.get('bio', "")
            picture_url = lens_data.get('picture', "")
            if isinstance(picture_url, dict):
                picture_url = picture_url.get('url', "")
    except Exception:
        pass

    return {
        'info': {
            'handle': row['handle'] if row['handle'] else "unknown",
            'display_name': row['display_name'],
            'bio': bio,
            'picture_url': picture_url,
            'owned_by': row['owned_by'],
            'created_on': str(row['created_on'])
        },
        'stats': [
            float(row['total_tips']),
            float(row['total_posts']),
            float(row['total_quotes']),
            float(row['total_reacted']),
            float(row['total_reactions']),
            float(row['total_reposts']),
            float(row['total_collects']),
            float(row['total_comments']),
            float(row['total_followers']),
            float(row['total_following'])
        ],
        'interactions': interaction_list
    }


def mock_bq_fetcher(profile_id: str) -> Dict[str, Any]:
    """
    Return mock data for testing without BigQuery access.
    
    Args:
        profile_id: The profile ID (used for display purposes).
        
    Returns:
        Dictionary containing mock 'info', 'stats', and 'interactions'.
    """
    return {
        'info': {
            'handle': 'orb_test_user',
            'display_name': 'Test User',
            'bio': 'Crypto enthusiast. Follow for updates.',
            'picture_url': '',
            'owned_by': '0x123abc456def789',
            'created_on': '2025-12-08 10:00:00'
        },
        'stats': [0, 5, 0, 0, 10, 2, 0, 1, 15, 10],
        'interactions': ['0x9259f5b38699acdc3f94714e683f38a511450904']
    }

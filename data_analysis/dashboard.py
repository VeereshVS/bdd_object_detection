"""Streamlit Dashboard for BDD100K Data Analysis.

Run with: streamlit run dashboard.py -- --data-root /path/to/bdd100k

This dashboard provides interactive visualization of the BDD100K
dataset analysis results for object detection.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.analyzer import BDD100KAnalyzer
from src.parser import BDD100KParser, DETECTION_CLASSES
from src.visualizer import (
    CLASS_COLORS,
    plot_attribute_distribution,
    plot_bbox_statistics,
    plot_class_distribution,
    plot_cooccurrence_matrix,
    plot_objects_per_image_histogram,
    plot_occlusion_truncation_stats,
    plot_spatial_heatmap,
    plot_train_val_comparison,
    visualize_sample,
)


def get_args():
    """Parse command line arguments for Streamlit."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=str,
        default="/data/bdd100k",
        help="Root directory of the BDD100K dataset",
    )
    # Filter out streamlit arguments
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args


@st.cache_resource
def load_data(data_root: str):
    """Load and parse the BDD100K dataset annotations."""
    parser = BDD100KParser(data_root)

    train_labels = str(
        Path(data_root) / "labels" / "bdd100k_labels_images_train.json"
    )
    val_labels = str(
        Path(data_root) / "labels" / "bdd100k_labels_images_val.json"
    )

    if Path(train_labels).exists():
        parser.parse_labels(train_labels, split="train")
    if Path(val_labels).exists():
        parser.parse_labels(val_labels, split="val")

    analyzer = BDD100KAnalyzer(parser)
    return parser, analyzer


def main():
    """Run the Streamlit dashboard."""
    st.set_page_config(
        page_title="BDD100K Object Detection Analysis",
        page_icon="🚗",
        layout="wide",
    )

    st.title("🚗 BDD100K Object Detection Data Analysis Dashboard")
    st.markdown("---")

    args = get_args()
    data_root = args.data_root

    # Sidebar configuration
    st.sidebar.header("Configuration")
    data_root_input = st.sidebar.text_input("Data Root", value=data_root)

    # Load data
    try:
        parser, analyzer = load_data(data_root_input)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.info("Please ensure the data path is correct and labels are available.")
        return

    available_splits = list(parser.annotations.keys())
    if not available_splits:
        st.error("No data loaded. Check the data path.")
        return

    # Navigation
    page = st.sidebar.selectbox(
        "Select Analysis",
        [
            "Overview",
            "Class Distribution",
            "Train vs Val Comparison",
            "Bounding Box Analysis",
            "Spatial Distribution",
            "Attribute Analysis",
            "Anomaly Detection",
            "Interesting Samples",
            "Co-occurrence Analysis",
        ],
    )

    if page == "Overview":
        st.header("Dataset Overview")
        cols = st.columns(len(available_splits))
        for i, split in enumerate(available_splits):
            with cols[i]:
                st.metric(
                    f"{split.capitalize()} Images",
                    len(parser.annotations[split]),
                )
                total_objects = sum(
                    ann.num_objects for ann in parser.annotations[split]
                )
                st.metric(f"{split.capitalize()} Objects", total_objects)

        st.subheader("Detection Classes")
        st.write(", ".join(DETECTION_CLASSES))

    elif page == "Class Distribution":
        st.header("Class Distribution Analysis")
        split = st.selectbox("Select Split", available_splits)
        dist_df = analyzer.analyze_class_distribution(split)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Distribution Table")
            st.dataframe(dist_df, use_container_width=True)
        with col2:
            st.subheader("Visualization")
            fig = plot_class_distribution(dist_df, title=f"{split.capitalize()} Set")
            st.pyplot(fig)

        # Objects per image
        st.subheader("Objects per Image Distribution")
        fig = plot_objects_per_image_histogram(parser.annotations[split])
        st.pyplot(fig)

    elif page == "Train vs Val Comparison":
        st.header("Train vs Validation Split Comparison")
        if "train" in available_splits and "val" in available_splits:
            comparison = analyzer.analyze_train_val_split()
            st.dataframe(comparison, use_container_width=True)
            fig = plot_train_val_comparison(comparison)
            st.pyplot(fig)
        else:
            st.warning("Both train and val splits are needed for comparison.")

    elif page == "Bounding Box Analysis":
        st.header("Bounding Box Statistics")
        split = st.selectbox("Select Split", available_splits)
        bbox_stats = parser.get_bbox_stats(split)
        fig = plot_bbox_statistics(bbox_stats)
        st.pyplot(fig)

        st.subheader("Detailed Statistics")
        stats_df = pd.DataFrame(bbox_stats).T
        st.dataframe(stats_df, use_container_width=True)

        # Occlusion/Truncation
        st.subheader("Occlusion & Truncation Rates")
        occ_stats = parser.get_occlusion_truncation_stats(split)
        if occ_stats:
            fig = plot_occlusion_truncation_stats(occ_stats)
            st.pyplot(fig)

    elif page == "Spatial Distribution":
        st.header("Spatial Distribution Heatmaps")
        split = st.selectbox("Select Split", available_splits)
        selected_class = st.selectbox("Select Class", DETECTION_CLASSES)
        fig = plot_spatial_heatmap(
            parser.annotations[split], selected_class
        )
        st.pyplot(fig)

    elif page == "Attribute Analysis":
        st.header("Analysis by Image Attributes")
        split = st.selectbox("Select Split", available_splits)

        attr_dist = parser.get_attribute_distribution(split)
        fig = plot_attribute_distribution(attr_dist)
        st.pyplot(fig)

        attr_analysis = analyzer.analyze_by_attributes(split)
        for attr_name, df in attr_analysis.items():
            st.subheader(f"Detections by {attr_name.capitalize()}")
            st.dataframe(df, use_container_width=True)

    elif page == "Anomaly Detection":
        st.header("Anomaly Detection")
        split = st.selectbox("Select Split", available_splits)
        anomalies = analyzer.detect_anomalies(split)

        st.subheader("Summary")
        summary = anomalies["summary"]
        cols = st.columns(3)
        cols[0].metric("High Density Images", summary["high_density_count"])
        cols[1].metric("Tiny Boxes", summary["tiny_boxes_count"])
        cols[2].metric("Invalid Boxes", summary["invalid_boxes_count"])

        cols2 = st.columns(3)
        cols2[0].metric("Huge Boxes", summary["huge_boxes_count"])
        cols2[1].metric("Extreme Aspect Ratios", summary["extreme_ar_count"])
        cols2[2].metric("Total Images", summary["total_images"])

        if anomalies["high_density_images"]:
            st.subheader("High Density Images (most objects)")
            st.dataframe(
                pd.DataFrame(anomalies["high_density_images"][:20]),
                use_container_width=True,
            )

    elif page == "Interesting Samples":
        st.header("Interesting/Unique Samples")
        split = st.selectbox("Select Split", available_splits)
        interesting = analyzer.find_interesting_samples(split, top_k=10)

        tab_names = list(interesting.keys())
        tabs = st.tabs(tab_names)
        for tab, name in zip(tabs, tab_names):
            with tab:
                samples = interesting[name]
                if samples:
                    st.dataframe(
                        pd.DataFrame(samples[:10]),
                        use_container_width=True,
                    )

    elif page == "Co-occurrence Analysis":
        st.header("Class Co-occurrence Matrix")
        split = st.selectbox("Select Split", available_splits)
        cooccurrence = analyzer.analyze_class_cooccurrence(split)
        fig = plot_cooccurrence_matrix(cooccurrence)
        st.pyplot(fig)

        st.subheader("Co-occurrence Table")
        st.dataframe(cooccurrence, use_container_width=True)


if __name__ == "__main__":
    main()

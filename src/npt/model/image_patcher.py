from itertools import cycle

import torch
import torch.nn as nn


class ImagePatcher(nn.Module):
    def __init__(
            self, dim_hidden, input_feature_dims, c, patcher_type):
        super(ImagePatcher, self).__init__()

        self.c = c

        # Global setting, avoids non-patching logic in NPT init
        self.model_image_patching = True

        D = len(input_feature_dims)  # Flattened size of an image

        # We reduce what the core model sees to a sequence of patches
        # (unordered, but with patch index embeddings), until the decoder
        self.image_n_patches = self.c.model_image_n_patches

        # Includes the target column
        self.num_input_features = self.image_n_patches + 1

        # Options: {'linear'}
        self.image_patch_type = self.c.model_image_patch_type

        # Share embedding weights across patches, or separate
        self.image_share_embed = self.c.model_image_share_embed

        # e.g. 3 for RGB
        self.image_n_channels = self.c.model_image_n_channels

        # If we use BERT augmentation, this must be 2, for
        # the continuous pixel intensity and the mask value.
        # Otherwise, this should be 1.
        self.dim_intensity = 1 + bool(self.c.model_bert_augmentation)
        self.dim_target_col = self.c.model_image_n_classes + bool(
            self.c.model_bert_augmentation)

        # Exclude target column (we assume it is right concatenated)
        image_input_shape = (D - 1, self.dim_intensity)

        # The number of patches must divide the number of pixels
        assert image_input_shape[0] % self.image_n_patches == 0

        # This is in raw intensities, i.e. counting each pixel in an
        # RGB image thrice
        self.patch_size = image_input_shape[0] // self.image_n_patches

        # Compute resizing constants
        n_features = len(input_feature_dims) - 1
        assert n_features % self.image_n_channels == 0

        if patcher_type == 'linear':
            # H = height, note that we are expecting square images for now
            # H = height = W = width
            flattened_image_size = n_features // self.image_n_channels
            self.image_H = int(flattened_image_size ** 0.5)
            assert flattened_image_size // self.image_H == self.image_H

            # Get number of rows of patches
            n_patches_per_side = self.image_n_patches ** 0.5
            assert int(n_patches_per_side) == n_patches_per_side
            n_patches_per_side = int(n_patches_per_side)

            # Get length of patches
            # (i.e. each patch is patch_side_length x patch_side_length)
            assert self.image_H % n_patches_per_side == 0
            self.patch_side_length = self.image_H // n_patches_per_side

        # ### Embeddings ###

        # Always use a linear out-embedding
        if self.image_share_embed:
            # Output into the number of intensities in a patch
            # (no mask dim needed), applied in a sliding fashion
            self.out_feature_embedding = nn.ModuleList([
                nn.Linear(dim_hidden, self.patch_size)])
        else:
            # Separate linear embedding for each patch
            self.out_feature_embedding = nn.ModuleList([
                nn.Linear(dim_hidden, self.patch_size)
                for _ in range(self.image_n_patches)])

        self.out_target_embedding = nn.Linear(
            dim_hidden, c.model_image_n_classes)

    def decode(self, X):
        # We receive a tensor of shape (N, n_patches + 1, E)

        # Feature Patch De-Embedding
        if self.image_share_embed:
            de_embeds = cycle(self.out_feature_embedding)
        else:
            de_embeds = self.out_feature_embedding

        X_ragged = []

        # Projects each batched feature patch of shape (N, E) to (N,
        for patch_index in range(X.shape[1] - 1):
            # X_patch.shape = (N, E)
            X_patch = X[:, patch_index, :]

            # de_embed.shape = (E, p) where p = patch size
            de_embed = next(de_embeds)

            # X_de_embed.shape = (N, p)
            X_de_embed = de_embed(X_patch)

            # Split into p columns of shape (N, 1)
            X_de_embed = torch.split(X_de_embed, 1, dim=1)
            X_ragged += X_de_embed

        # Append projection of target column
        X_ragged.append(self.out_target_embedding(X[:, -1, :]))

        return X_ragged

    def get_npt_attrs(self):
        """Send a few key attributes back to the main model."""
        return {'num_input_features': self.num_input_features,
                'image_n_patches': self.image_n_patches,
                'patch_size': self.patch_size}

    def preprocess_flattened_image(self, X_ragged):
        """
        Prior to applying the Linear transforms, we wish to reshape
        our features, which constitute the image:
            * D = total number of columns (including the target)
            (N, D - 1, dim_intensity)
            where dim_intensity is 2 if we are using masking, 1 otherwise
            to (N, (D - 1) // n_channels, dim_intensity * n_channels)

        This is necessary because, e.g., CIFAR-10 flattens images to be of
            format 1024 R, 1024 G, 1024 B. We must reshape to make sure
            the patching has the correct receptive fields.

        Returns:
            Reshaped X_features, X_target column
        """
        # Shape (N, D - 1, dim_intensity)
        # where dim_intensity = 2 if we have continuous pixel intensity + mask
        # or 1 if we just have the pixel intensity (no BERT augmentation mask)
        X_features = torch.stack(X_ragged[:-1], 1)

        # Reshape to (N, (D - 1) // n_channels, dim_intensity * n_channels)
        X_features = torch.reshape(
            X_features,
            (X_features.size(0),
             X_features.size(1) // self.image_n_channels,
             self.dim_intensity * self.image_n_channels))

        # Shape (N, 1, H_j) where H_j = num_categories + bool(is_mask)
        # (e.g. 2, for image regression with BERT augmentation)
        X_target = X_ragged[-1]

        return X_features, X_target


class LinearImagePatcher(ImagePatcher):
    def __init__(self, input_feature_dims, dim_hidden, c):
        super(LinearImagePatcher, self).__init__(
            dim_hidden, input_feature_dims, c, patcher_type='linear')

        self.patch_n_pixels = self.patch_side_length * self.patch_side_length
        pixel_input_dims = self.dim_intensity * self.image_n_channels

        # Each patch embedding should be shape
        # (patch_n_pixels, (1 + bool(is_mask)) * n_channels, dim_feature_embedding)
        if self.image_share_embed:
            self.in_feature_embedding = nn.ParameterList([
                nn.Parameter(torch.empty(
                    self.patch_n_pixels, pixel_input_dims,
                    dim_hidden))])
        else:
            self.in_feature_embedding = nn.ParameterList([
                nn.Parameter(torch.empty(
                    self.patch_n_pixels, pixel_input_dims,
                    dim_hidden))
                for _ in range(self.image_n_patches)])

        for embed in self.in_feature_embedding:
            nn.init.xavier_uniform_(embed)

        self.in_target_embedding = nn.Linear(
            self.dim_target_col, dim_hidden)

    def encode(self, X_ragged):
        # Feature Patch Embedding
        # Embed to a list of n_patch tensors,
        # each of size (N, dim_feature_embedding)

        X_features, X_target = self.preprocess_flattened_image(X_ragged)

        if self.image_share_embed:
            embeds = cycle(self.in_feature_embedding)
        else:
            embeds = self.in_feature_embedding

        X_embeds = []
        for pixel_index in range(0, X_features.shape[1], self.patch_n_pixels):
            # Projection:
            # n: batch dimension, number of rows
            # p: patch size in number of locations (e.g., num RGB pixels)
            # h: dim_intensity * n_channels
            #       = (1 + 1) * n_channels if we use BERT masking,
            #       = 1 * n_channels otherwise
            # e: dim_feature_embedding, NPT hidden dimensions

            # X_input.shape = (n, p, h)
            X_input = X_features[
                :, pixel_index:pixel_index+self.patch_n_pixels, :]

            # embed.shape = (p, h, e)
            embed = next(embeds)

            X_embeds.append(torch.einsum('nph,phe->ne', X_input, embed))

        X_embeds.append(self.in_target_embedding(X_target))
        X_embed = torch.stack(X_embeds, 1)

        return X_embed

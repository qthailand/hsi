"""The list of feature modules shown in the HSI *Modules* menu.

Adding a module to the application = writing an :class:`~windows.integrations.Integration`
for it and appending it here; the main window needs no change.
"""
from __future__ import annotations

from typing import List

from .integrations import (Integration, RadiometricIntegration, ReferenceROIIntegration,
                           SmileIntegration, SpectralResolutionIntegration, WavelengthIntegration)

#: Integration classes in menu order.
INTEGRATIONS = (RadiometricIntegration, SmileIntegration, WavelengthIntegration,
                SpectralResolutionIntegration, ReferenceROIIntegration)


def default_integrations(host) -> List[Integration]:
    """Instantiate every registered integration for ``host`` (see :mod:`windows.integrations`)."""
    return [integration_class(host) for integration_class in INTEGRATIONS]

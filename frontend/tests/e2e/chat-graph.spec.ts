import { test, expect, Page } from '@playwright/test';

const CHAT_RESPONSE = {
  question: 'Who designed the Analytical Engine?',
  answer: 'Charles Babbage designed the Analytical Engine in 1837.',
  citations: [
    {
      chunk_id: 'chunk-1',
      document_id: 'doc-ada',
      document_name: 'ada_lovelace.pdf',
      page_number: 1,
      snippet: 'Charles Babbage designed the Analytical Engine in 1837.',
    },
  ],
  evidence: [
    {
      chunk_id: 'chunk-1',
      document_id: 'doc-ada',
      document_name: 'ada_lovelace.pdf',
      page_number: 1,
      snippet: 'Charles Babbage designed the Analytical Engine in 1837.',
      source: 'vector',
    },
  ],
  has_evidence: true,
  retrieval_metadata: { vector_chunks: 1, graph_chunks: 0 },
};

const GRAPH_RESPONSE = {
  nodes: [
    { id: 'person:babbage', label: 'Charles Babbage', type: 'PERSON' },
    { id: 'person:lovelace', label: 'Ada Lovelace', type: 'PERSON' },
    { id: 'engine:analytical', label: 'Analytical Engine', type: 'OBJECT' },
  ],
  edges: [
    {
      id: 'rel:1',
      source: 'person:babbage',
      target: 'engine:analytical',
      label: 'DESIGNED',
      confidence: 0.95,
    },
    {
      id: 'rel:2',
      source: 'person:lovelace',
      target: 'engine:analytical',
      label: 'WROTE_ABOUT',
      confidence: 0.9,
    },
  ],
  total_nodes: 3,
  total_edges: 2,
};

async function openChat(page: Page) {
  await page.goto('/chat');
  await expect(page.getByRole('heading', { name: 'AI Chat' })).toBeVisible();
}

test.describe('chat display contract', () => {
  test('renders the answer and cited source returned by the API', async ({ page }) => {
    await page.route('**/api/v1/chat/', async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(CHAT_RESPONSE),
      });
    });

    await openChat(page);

    await page.getByPlaceholder('Ask a question...').fill(CHAT_RESPONSE.question);
    await page.getByRole('button', { name: 'Send' }).click();

    await expect(page.getByText(CHAT_RESPONSE.answer)).toBeVisible();
    await expect(page.getByText('Sources:', { exact: true })).toBeVisible();
    await expect(page.getByText('ada_lovelace.pdf - Page 1', { exact: true })).toBeVisible();
  });

  test('sends the typed question to the chat endpoint', async ({ page }) => {
    let requestBody: { question?: string; document_id?: string } | undefined;

    await page.route('**/api/v1/chat/', async route => {
      requestBody = route.request().postDataJSON();
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(CHAT_RESPONSE),
      });
    });

    await openChat(page);
    await page.getByPlaceholder('Ask a question...').fill('  Who designed the Analytical Engine?  ');
    await page.getByRole('button', { name: 'Send' }).click();

    await expect(page.getByText(CHAT_RESPONSE.answer)).toBeVisible();
    expect(requestBody?.question).toBe('  Who designed the Analytical Engine?  ');
    expect(requestBody?.document_id).toBeUndefined();
  });
});

test.describe('graph display contract', () => {
  test('renders graph nodes, edges, and the node details panel', async ({ page }) => {
    await page.route(/\/api\/v1\/graph(?:\/|\?|$)/, async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify(GRAPH_RESPONSE),
      });
    });

    await page.goto('/graph');
    await expect(page.getByRole('heading', { name: 'Graph Explorer' })).toBeVisible();

    const node = page.locator('.react-flow__node', { hasText: 'Ada Lovelace' });
    await expect(node).toBeVisible();

    // Edge labels are rendered by React Flow in the edge layer.
    await expect(page.getByText('DESIGNED', { exact: true })).toBeVisible();

    await node.click();
    await expect(page.getByRole('heading', { name: 'Node Details' })).toBeVisible();
    await expect(page.getByText('Label: Ada Lovelace')).toBeVisible();
    await expect(page.getByText('Type: PERSON')).toBeVisible();
    await expect(page.getByText('ID: person:lovelace')).toBeVisible();
  });

  test('shows an empty state when the graph has no nodes', async ({ page }) => {
    await page.route(/\/api\/v1\/graph(?:\/|\?|$)/, async route => {
      await route.fulfill({
        status: 200,
        contentType: 'application/json',
        body: JSON.stringify({ nodes: [], edges: [], total_nodes: 0, total_edges: 0 }),
      });
    });

    await page.goto('/graph');
    await expect(page.getByText('No graph data available')).toBeVisible();
  });

  test('shows an error state when the graph API fails', async ({ page }) => {
    await page.route(/\/api\/v1\/graph(?:\/|\?|$)/, async route => {
      await route.fulfill({ status: 500, body: 'boom' });
    });

    await page.goto('/graph');
    // The page currently logs the failure and falls back to its empty state.
    await expect(page.getByText('No graph data available')).toBeVisible();
  });
});
